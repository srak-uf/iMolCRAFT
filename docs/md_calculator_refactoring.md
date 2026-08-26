# MD サンプリングの MDCalculator 化と自己完結チェックポイント

作業日: 2026-08-25 〜 2026-08-26 / ブランチ `dev`

## 背景と経緯

`dmff_utils.md_sample` に対して一連の変更を行った。

1. **ログ出力先の選択** — 標準出力固定だったものを `md_log` (`"stdout"` / `"file"` /
   `"none"`) で切り替え可能に。コミット `a70a03d`
2. **任意アニールスケジュール** — `anneal_Tmax` + `anneal_totalsteps`（降温のみ、
   しかも Tmax へ瞬間ジャンプ）を廃止し、温度点リスト `anneal_T` と各レグの MD
   ステップ数リスト `anneal_steps`、更新間隔 `anneal_interval` に置換。コミット
   `5e9049e`
3. **整合性チェックと引数削減** — parser / trainer / example / tests の追従、
   トレーナー必須キーの削減。`_SAMPLING_FIELDS` / `_SAMPLING_KEYS` /
   `_MD_KEYS` の変換表 3 つを撤去し、直接展開と名前統一に置き換えた
4. **MDCalculator 化と自己完結チェックポイント**（本ドキュメントの主題）

## 1. ログ出力先 (`md_log`)

`MDCalculator` の設定 `md_log` / `md_logfile`。`"none"` では
`StateDataReporter` 自体を attach しない。`"file"` の既定パスは
`mdlogs/<trajectory の stem>.log` で、レプリカごとに別ファイルになる。
`md_logfile` を明示するとレプリカ全部が同じファイルを上書きする点に注意。

`ThermodynamicTrainer.__init__` の `md_log` / `md_logfile` から制御する。

## 2. アニールスケジュール

```yaml
anneal_T: [300.0, 500.0, 300.0]   # 温度点（K）
anneal_steps: [50000, 100000]     # 各レグの MD ステップ数。anneal_T より 1 つ少ない
anneal_interval: 100              # 設定温度を更新する間隔（MD ステップ）
```

- 300 K → 500 K を 50000 ステップで昇温、500 K → 300 K を 100000 ステップで降温
- `anneal_T` / `anneal_steps` は **list か tuple のみ**。裸の数値は `TypeError`
  （文字列も弾く）
- 片方でも None / 空ならアニールなし
- 各レグは終端温度にちょうど着地する（`interval` が割り切れない場合は最後の
  チャンクを短くする）

### `anneal_interval` の根拠（実測）

温度更新の粒度によるオーバーヘッドは実質ゼロだった（CPU platform, N=20000）。

| 更新間隔 | スループット |
|---|---|
| 更新なし | 2975 step/s |
| 100 step ごと | 2955 step/s |
| 1 step ごと | 2978 step/s |

差はノイズレベル。OpenMM に組み込みのアニールスケジューラは無く、積分器内で
自動化するには `CustomIntegrator` に global variable を持たせて LangevinMiddle を
自前で書き直す必要がある（HBonds 拘束も手で入れる）。実測上ペナルティが無いので
Python ループで十分と判断した。

**OpenCL platform での再測定は未実施**（バックグラウンド実行が前セッション終了で
中断した）。GPU で差が出る可能性は残っている。既定の `interval=100` は
その保険でもある。

## 3. トレーナーの必須キー削減

`sampling_params` のうち **トレーナー自身が読むものだけ** を必須にした。

必須:
`init_structure` / `temperature_K` / `rcut_nm` / `nonbondedmethod` /
`ensemble` / `neff`

省略可:
- `pressure_bar` → PV 項なし（0.0）。**NPT系では必須**、無ければ明示エラー
- `dispcorr` → False
- MD 設定 7 個（`dt_fs` `nstxout` `relax_steps` `prod_steps` `anneal_*`）
  → `MDCalculator.SETTINGS` の既定値

NVT で `pressure_bar` が不要な根拠: `buildInputEnergyFunction` は NVT で
`ensemble_cns = 0.0` として pressure を 0 倍する。`OpenMMSampleState` は PV 項を
無条件に足すが、NVT は体積一定なので全フレーム共通の定数となり、MBAR では
状態ごとの定数シフトが自由エネルギー項に吸収されて重みに影響しない。

**挙動変更**: NVT の `P_bar` 既定値が 1.0 → 0.0。重み・loss・neff は不変だが
`utarget` の絶対値は定数分ずれる。NPT の YAML で `pressure_bar` 未記載は
エラーになる（従来は parser が黙って 1 bar を注入していた）。

## 4. MDCalculator 化（本題）

### ファイル構成

`imolcraft/calculator/md.py`（新規）に以下を移した:

- `VALID_ENSEMBLES` / `MD_LOG_MODES` / `NONBONDED_METHODS`
- `resolve_nonbondedmethod`（trainer にあった `_resolve_nonbondedmethod` を統合）
- `_open_md_log` / `_anneal_schedule` / `_ramp_temperature` / `_make_barostat`
- `class MDCalculator`
- `md_sample`（薄いラッパとして残置）

依存の向きは `trainer.dmff_utils` → `calculator.md`。`parser_dmffyaml` が
`VALID_ENSEMBLES` を使うため dmff_utils が import する。import コストの追加は
ゼロ（calculator パッケージは既に読み込まれている）。

### 設計

状態の定義は構築時、力場と出力トラジェクトリは `run()` の引数。

```python
calc = MDCalculator("supercell_bonds.pdb", temperature_K=300, ensemble="anisonpt",
                    anneal_T=[300, 500, 300], anneal_steps=[50000, 100000])
xtc = calc.run(ffxml="ff.xml", trajectory="sample_0.xtc")
record = calc.to_dict()             # 既定値込みの完全なレシピ
calc2 = MDCalculator.from_dict(record)
```

設定名と既定値は `MDCalculator.SETTINGS`（dict）**一箇所だけ** に置き、
`__init__(self, init_structure, **settings)` がそれを展開する。これにより
`to_dict()` が「省略された設定も既定値込みで」書き出せる。未知のキーは
`TypeError`（打ち間違い検出）。

**設定名は YAML の sampling キーと一致させてある**（`temperature_K` / `rcut_nm` /
`dt_fs` / `dispcorr` / `init_structure`）。当初はトレーナー側に変換表 `_MD_KEYS`
を置いていたが、13 エントリ中 8 個が恒等写像で無駄だったため、名前を YAML 側に
揃えて表ごと削除した。トレーナーは

```python
settings = {k: v for k, v in params.items() if k in MDCalculator.SETTINGS}
```

で MD 設定だけを拾う。`neff` / `pressure_bar` は `SETTINGS` に無いので自然に
残る。`device` / `md_log` / `md_logfile` は run 単位の設定なので、その後に
トレーナー側の値で上書きする。

なお `useHbondConstraint` / `rigidWater` は YAML に対応キーが無く、OpenMM の
`createSystem` 側の名前をそのまま使っている。

### 自己完結チェックポイント

`write_checkpoint` に追加したキー:

| キー | 内容 |
|---|---|
| `md_params` | `[c.to_dict() for c in self.md_calculators]` |
| `restart_args` | `ffxml_list` / `nums_ffxml` / `pdbfile` / `device` / `md_log` / `md_logfile` / `resample_freq` |
| `initial_ffxml` | `chkpoint_<label>.xml` |
| `resample_counter` | 再サンプリングカウンタ |
| `loss_fn` | pickle 可能なら記録、不可なら None（`_picklable` ヘルパ） |

削除したキー（`sampling_params` と完全重複、`from_checkpoint` は読み返して
いなかった）: `T_K` `P_bar` `anneal_*` `relax_steps` `rc_nm` `prod_steps`
`nstxout` `neff` `dt_fs` `ensemble`

`from_checkpoint` は全引数 optional になり、pkl パスだけで復元できる:

```python
trainer = ThermodynamicTrainer.from_checkpoint("train_state_ff_opt.pkl")
trainer.fit(500, 2)
```

- 内部で `setup()` を呼ぶ（MBAR estimator を作るため。呼ばないと `fit` が
  `AttributeError: estimator`）。`setup=False` で抑止可
- `setup()` は `opt_state` を初期化するので、**その後に** 記録された
  `opt_state` / `_epoch` / 履歴を書き戻す順序になっている
- MD は `sampling_params` から再導出せず `md_params` から直接復元する。
  既定値が将来変わってもサンプリング内容が変わらない
- `device` / `md_log` / `md_logfile` は run 単位の設定なので、復元した
  calculator にトレーナー側の値を上書きする（引数での override が効く）

### 復元できないもの

- lambda で書かれた `loss_fn`（pickle 不可、None として記録し引数を要求）
- `add_modifyfn` で登録した関数（記録しない。restart 後に再登録が必要）

## 既知の問題（本作業とは無関係、未修正）

`tests/test_trainer/test_trainer.py::TestDistanceTrainer::test_save_load` が
常時失敗する。`BaseTrainer.from_checkpoint`（[base.py:389](../imolcraft/trainer/base.py#L389)）
が全トレーナー共通で `trainer.validation_pred` を参照するが、この属性は
`ThermodynamicTrainer.__init__` でしか初期化されない。`DistanceTrainer` で
`AttributeError`。commit `2f05d3b`（validation 対応）の混入で、`16f7cfd` の
ワークツリーでも同じ失敗を確認済み。

修正案: `BaseTrainer.__init__` で初期化するか `getattr` 経由にする。

## テスト

- `tests/test_calculator/test_md.py`（新規, 32 件）— アニールスケジュール展開・
  型チェック・温度ランプ・ログ出力先・`MDCalculator` の to_dict/from_dict・
  設定名が sampling キーと一致すること・barostat
- `tests/test_trainer/` — 移動したテストを除去し整理

ベースライン: `tests/test_trainer tests/test_calculator` → **214 passed / 1 failed**
（失敗は上記の既知問題のみ）

実行には `conda activate imc_cpu` が必要。
