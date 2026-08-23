# trainer リファクタリング記録

対象: `imolcraft/trainer/`（loss.py / base.py / dmff_utils.py / trainer.py、計 2511 行）

## 方針
crafter・analyzer・io・calculator と同じ。ファイル構成は維持、挙動は完全維持
（等価変換のみ）、潜在バグは修正せず記録する。

## 検証方法
学習ループ本体は MD と QM を伴うため直接は回せない。旧コード（git HEAD 版）を別モジュール
としてロードし、外部実行を伴わない全経路を突き合わせた。

- **17 項目一致**
  - `mse_energy` を 2 サイズ × 重み 2 種 × zeropoint 3 種 × 正規化 2 種の 24 通りで
    （`atol=0`、つまりビット単位）
  - `wrightfactor` / `jsdivergence`
  - `neutralize` を 4 通りの拘束条件で
  - `get_rescharges_from_residues` / `get_chgparams_from_rescharges` /
    `update_rescharges_from_params` / `update_ffinfo_from_params` /
    `update_ffinfo_from_rescharges` / `vsiteinfo_to_params` を 3 つの実 ffxml
    （ethane、仮想サイト 2 個、仮想サイト 1 個）× ratio 2 通りで
  - `parser_dmffyaml` の戻り値と、書き出す `_parsed.yaml` が**バイト一致**
  - 検証エラー 4 種（不正な target 名 / 不正な ensemble / weight 欠落 / rdf のキー欠落）の
    例外型とメッセージが一致
- `pytest tests -m "not qm"` は 221 passed / 10 failed / 1 error。失敗と error は
  packmol / antechamber / parmchk2 / psi4 の未インストールによる環境要因で、
  リファクタ前と同じ顔ぶれ。

## 主な変更

### base.py（588 -> 548 行）
最大の重複は、`BaseTrainer.__init__` と `SumTrainer.__init__` に**丸ごと同じ 28 行**が
書かれていたオプティマイザ構築だった。しかもその中の if/else は

```python
if opt_fftype == "NonbondedForce/charge":
    multiTrans[opt_fftype] = genOptimizer(..., nonzero=False)
else:
    multiTrans[opt_fftype] = genOptimizer(..., nonzero=False)  # Should be True
```

と**両分岐が完全に同一**で、条件が意味を持っていなかった。`# Should be True` は
「本当は else 側を True にしたい」という書き残しなので、コメントごと残した上で分岐を畳んだ。

| 抽出したヘルパー | 置き換えた内容 |
| --- | --- |
| `_build_optimizer` | 上記の 28 行 × 2。整数パラメータをマスクする理由も docstring に記載 |
| `_broadcast_lr_clip` | lr / clip をスカラーからリストへ広げる処理 × 2（assert を ValueError に） |
| `_nan_recovery_gradients` | 損失が NaN / Inf のときのランダム摂動 × 4 |
| `_print_memory` | `psutil.Process(os.getpid())` のメモリ表示 × 5 |
| `plot_learning_curve` | 学習曲線 3 枚（線形 / log-y / log-log）の描画。base.py と trainer.py にまたがる重複 |
| `SumTrainer._substep` | 2 つの sub-trainer に対する同一処理 |
| `SumTrainer._scatter_to_subtrainers` / `_gather_from_subtrainers` | 結合パラメータの分配と再結合 × 4 |

`app.PDBFile.writeFile(..., open(...))` の閉じ忘れも `with` にした。

### loss.py（193 -> 269 行、増分は docstring）
- `_squared_error` 抽出: zeropoint の 3 分岐を分離。
- `_mbar_weights` 抽出: MBAR の重み計算。estimator が入力エネルギーを持つかで
  経路が分かれる理由を docstring に記載。
- `_charge_penalty_loss` 抽出。
- `_DISTRIBUTION_LOSSES` 辞書化: `wrightfactor` / `jsdivergence` の if-elif。
- 可変デフォルト引数 `charge_penalty={...}` を `None` + `_NO_CHARGE_PENALTY` 定数に。
- 型注釈 `efunc: any`（組み込み関数）を `Any` に。
- 未使用 import 6 個（`app` / `unit` / `Hamiltonian` / `DMFFTopology` / `jit` /
  `update_ffinfo_from_params`）を削除。
- マジックリテラルを `IMPLEMENTED_WEIGHT_SCHEMES` / `_NPT_ENSEMBLES` /
  `_SCALAR_TARGETS` / `_DISTRIBUTION_TARGETS` に。

### dmff_utils.py（761 -> 775 行、増分は docstring と定数）
- `parser_dmffyaml` の目標値検証を `REQUIRED_TARGET_KEYS` テーブル駆動に。
  同じ形の `if "xxx" not in ...: raise KeyError(...)` が 11 個並んでいた。
- `_make_barostat` 抽出: アンサンブル別の圧力制御。3 種の NPT が「箱のどこまで動かすか」で
  違うことを docstring に記載。
- `md_sample` の `createSystem` 2 重複（`constraints` の有無だけが違う）を kwargs に統一。
  `if useDispersionCorrection: setUseDispersionCorrection(True) else False` も直接代入に。
- `plot_compare` の `ax[i_plot // 2, i_plot % 2]` が 10 回書かれていたのを変数に。
- `assert` 4 箇所を `ValueError` に（`python -O` で消えるため）。実際の長さを
  メッセージに含めた。

### trainer.py（969 -> 930 行）
`DistanceTrainer` と `DihedralTrainer` の `get_loss_gradients` が**完全に同一**、
`after_step` の前半も**完全に同一**だった。`_ScanTrainerMixin` を作って集約した。

| 抽出したヘルパー | 置き換えた内容 |
| --- | --- |
| `_ScanTrainerMixin.get_loss_gradients` | 2 クラスで同一だった 20 行 |
| `_ScanTrainerMixin._push_params_to_ff` | `after_step` 前半の 15 行 × 2 |
| `_ffparams_without_charge` | 電荷と仮想サイト力を除く処理。なぜ除くのかを docstring に記載 |
| `_scan_positions_nm` | angstrom -> nm 変換と仮想サイト挿入 × 4 |
| `_neighbour_pairs` | 近傍リスト構築 × 2 |
| `_qm_energies` | 参照エネルギーの積み上げ × 3 |
| `_sum_loss_and_grads` | 損失と勾配の累積 |
| `_resolve_nonbondedmethod` | PME / LJPME の解決 × 3（`raise AssertionError` を `ValueError` に） |
| `_SAMPLING_FIELDS` テーブル | 14 個の `self.xxx.append(cast(sampling_param["yyy"]))` が並んだ 30 行 |
| `ThermodynamicTrainer._run_md` / `_add_sample` | `setup` と `_resample` に重複していた MD 実行と MBAR 状態登録（各 35 行） |

`raise AssertionError` 3 箇所を `ValueError` にし、`app.PDBFile.writeFile(..., open(...))` を
`with` にした。学習曲線の描画は base.py の `plot_learning_curve` に委譲。

## テストの新設

`tests/test_trainer/test_trainer_pure.py` を新設し、外部プログラムや長時間の MD を
必要としない 35 件を追加した。既存の `test_trainer.py` は psi4 を要求するテストを含み、
このマシンでは常に error になる。

- `loss.py`: `_squared_error` の 3 方式（qmmin が QM 最小点で 0 になること等）、
  同一エネルギーで `mse_energy` が 0 になること、未知の重み方式で `ValueError`、
  同一分布で `wrightfactor` / `jsdivergence` が 0 になること
- `dmff_utils.py`: `neutralize` が指定した正味電荷に到達すること、拘束グループを守ること、
  長さ不一致で `ValueError`、`_make_barostat` の 5 アンサンブル、目標キー表の整合
- `base.py`: `_broadcast_lr_clip` の展開と長さ検証、`_nan_recovery_gradients` が
  決定的かつ十分小さいこと
- `trainer.py`: `_ffparams_without_charge` / `_resolve_nonbondedmethod` /
  `_qm_energies` / `_sum_loss_and_grads`

なお `tests/test_calculator/test_pure_functions.py` と同名になり pytest が
「import file mismatch」で collection error を起こしたため、両者を
`test_calculator_pure.py` / `test_trainer_pure.py` に改名した（テスト用ディレクトリに
`__init__.py` が無いため、basename が衝突すると同一モジュール扱いになる）。

## 未修正の潜在バグ・気になる点

1. `loss.mse_energy` の `weight_scheme="nonboltzmann"` は `IMPLEMENTED_WEIGHT_SCHEMES`
   に載っているが実装が `pass` のみで、`weight` が未定義のまま参照され
   `UnboundLocalError` になる。テスト側も `# , "nonboltzmann"` とコメントアウトされており、
   既知の未実装。
2. `dmff_utils._get_charges_types` は参照 0 件。

## 修正済みバグ / 追加機能: SumTrainer の再開（2026-08-21）

### 背景

`SumTrainer` を pickle から再開できるか調べたところ、3 つの欠落が見つかった。

| クラス | 修正前の `write_checkpoint` | 修正前の `from_checkpoint` |
| --- | --- | --- |
| `DistanceTrainer` | 独自（正しく dump） | あり |
| `DihedralTrainer` | **基底の no-op（何も保存しない）** | **なし** |
| `ThermodynamicTrainer` | 独自（正しく dump） | あり |
| `SumTrainer` | 独自だが **`pickle.dump` を呼んでいない** | **なし** |

加えて `SumTrainer.__init__` は `restart_xml` を引数に取りながら本体で一度も参照しておらず、
xml 経由の再開もできなかった。

### 「sub の pickle 2 つ」では足りない理由

パラメータを実際に更新しているのは `SumTrainer` 自身の `optimizer` / `opt_state` で、
sub 側の `opt_state` は別物。しかも `_substep` は sub の `optimizer.update` を呼んで
状態だけ進め、`updates` は捨てている（親側でまとめて適用するため）。つまり sub の
`opt_state` は「適用されていない更新の履歴」であって、親の状態とは対応しない。

Adam なら 1 次・2 次モーメントとステップ数を保持している。ここが失われると、再開直後に
バイアス補正がやり直しになって実効ステップ幅が跳ね、パラメータごとのスケーリングも消える。

| SumTrainer の状態 | sub の pickle から復元できるか |
| --- | --- |
| `ffparams` | 連結すれば可 |
| `ffparams_mapping` | 形状から再構築可 |
| **`opt_state`** | **不可（親固有）** |
| `losses` | `w0*l1 + w1*l2` で再計算可 |
| `epochs` / `_epoch` | 可 |
| `weight` | コンストラクタ引数 |
| `best_params` / `best_epoch` / `best_loss` | 不可（誰も保存していない。全 trainer 共通の欠落） |

結論として **sub の pickle と SumTrainer 専用の pickle の両方が要る**。sub 側は
`calculator` / `inputs` / `estimator` など親が複製していない状態を持ち、親側は
`opt_state` を持つ。

### 実装

1. **`SumTrainer.write_checkpoint` に `pickle.dump` を追加**（バグ修正）。`dump_dict` に
   `weight` を追加して自己完結させた。
2. **`DihedralTrainer` に `write_checkpoint` / `from_checkpoint` を追加**。
   `DistanceTrainer` とほぼ同形だが、ファイル名を `train_state_{label}.pkl` にした。
   `DistanceTrainer` は `train_state.pkl` 固定で、`SumTrainer` が両方の
   `write_checkpoint` を呼ぶと衝突しうるため（下記の未修正 1 番）。
3. **`SumTrainer.from_checkpoint(ckpt, trainer1, trainer2, ...)` を追加**。復元済みの
   sub インスタンスを受け取り、親の `ffparams` / `opt_state` / `_epoch` / `losses` /
   `epochs` を上書きしてから `_scatter_to_subtrainers()` で sub へ配り直す。
4. **`SumTrainer.__init__` が `restart_xml` を尊重するように**（基底と同じく
   `merge_xml` の結果を差し替える）。

あわせて `SumTrainer.setup()` が復元済みの `opt_state` を潰さないようにした。

```python
if getattr(self, "opt_state", None) is None:
    self.opt_state = self.optimizer.init(self.ffparams)
```

新規インスタンスは `opt_state` を持たないので従来どおり初期化され、復元したものだけが
自分の状態を保つ。

### 検証

MD も QM も走らせないダミー sub-trainer で往復を確認した。

```
checkpoint: train_state_t1_t2.pkl  size=6104 bytes
保存されたキー: ['clip','epoch','epochs','ffinfo','ffparams','label','losses',
                'lr','opt_fftypes','opt_state','optimizer_algo','weight']
ffparams 一致 : True
opt_state 一致: True   <- setup() で潰されない
losses 一致   : True
weight 復元   : [1.0, 2.0]
sub へ配布済み: True
再開後の epoch: 4
```

回帰テストを 4 件追加した（空でない pkl が書かれること、`opt_state` が復元され
`setup()` で潰されないこと、復元後に学習を続けられること、4 クラスすべてが
独自の checkpoint メソッドを持つこと）。


## 出力ファイル名の統一（2026-08-21）

`SumTrainer` は最後に両 sub-trainer の `write_checkpoint` を呼ぶため、sub が固定名で
書くと互いに上書きしてしまう。`DistanceTrainer` だけが label を含まない名前だったので、
他の 3 クラスに揃えた。

| | 変更前 | 変更後 |
| --- | --- | --- |
| `DistanceTrainer` の pkl | `train_state.pkl` | `train_state_{label}.pkl` |
| `DistanceTrainer` の周期 xml | `loop-{epoch}.xml`（カレント直下） | `xmlfiles/loop_{label}-{epoch}.xml` |
| `DihedralTrainer` の周期 xml | `xmlfiles/loop-{epoch}.xml` | `xmlfiles/loop_{label}-{epoch}.xml` |

周期 xml は「書いた直後に自分で読み直す」だけで外部からは参照されない（リポジトリ全体を
走査して確認済み）。2 箇所に散っていた「ディレクトリ作成 → renderXML → パスを返す」を
`_ScanTrainerMixin._render_loop_xml` にまとめた。

pkl の改名は破壊的変更にあたる。`from_checkpoint` はパスを引数で受け取るので読む側の
コードは変更不要だが、`trainer_checkpoint="train_state.pkl"` と固定名で書いている
呼び出しは影響を受ける。リポジトリ内では `tests/test_trainer/test_trainer.py` の 1 箇所が
該当したので `f"train_state_{trainer.label}.pkl"` に更新した。
`docs/source/thermodyn_perturb.md` の例は元から `ThermodynamicTrainer` のもので、
本文が `train_state_....pkl` と書いているとおり label 入りが前提なので影響しない。

回帰テストを 2 件追加した（4 クラスの `write_checkpoint` が
`f"train_state_{self.label}.pkl"` を使い固定名を使わないこと、周期 xml が
`_render_loop_xml` 経由で label 入りになること）。

## best_params / best_epoch / best_loss の保存（2026-08-21）

`BaseTrainer.fit` は損失が更新されるたびに最良のスナップショットを記録する。

```python
if len(self.losses) == 0 or self.loss < min(self.losses):
    self.best_params = self.ffparams
    self.best_epoch = self._epoch
    self.best_loss = self.loss
    self.ff.renderXML(f"{self.label}_best.xml")
```

しかしこの 3 つはどのクラスの checkpoint にも入っていなかった。再開すると `losses` は
復元されるので `min(self.losses)` は正しいが、`best_params` は None のままになり、
過去の最良より良い値が出るまで「最良のパラメータ」を失った状態で走ることになる。
さらに `fit` を回す前は属性自体が存在せず、`trainer.best_params` が `AttributeError`
だった。

対応は 3 つ。

1. `BaseTrainer.__init__` と `SumTrainer.__init__` で `best_params` / `best_epoch` /
   `best_loss` を None に初期化し、属性が常に存在するようにした。
2. `_best_checkpoint_fields()` / `_restore_best(dump_dict)` を `BaseTrainer` に追加し、
   4 クラスの `write_checkpoint` / `from_checkpoint` から使うようにした。
   古い checkpoint には該当キーが無いので `dict.get` で None にフォールバックする。
3. 回帰テストを 3 件追加（学習前に属性が存在すること、再開後も最良が保たれること、
   4 クラスすべてが保存と復元を行うこと）。

## ThermodynamicTrainer.from_checkpoint の attr_lists ループ削除（2026-08-21）

```python
attr_lists = ["epoch", "epochs", "losses", "ff_info"]
for key, value in dump_dict.items():
    if key in attr_lists:
        setattr(trainer, key, value)
```

2 つの問題があった。

- `"ff_info"` は保存側のキー（`"ffinfo"`）と綴りが違うため一度も一致しない。
- 仮に `"ffinfo"` に直しても `setattr(trainer, "ffinfo", ...)` は `trainer.ffinfo` という
  未使用の属性を作るだけで、実際に力場を保持している `trainer.ff.ffinfo` には届かない。

`"epoch"` も同様に `trainer.epoch`（未使用）を作るだけで、直後の
`trainer._epoch = dump_dict["epoch"]` が正しい代入をしている。結局このループで意味が
あったのは `"epochs"` と `"losses"` の 2 つだけだった。

`ThermodynamicTrainer` は `initial_ffxml` を必須引数に取り `restart_xml` として基底へ渡す
設計で、`ffparams` も pickle から復元していない。つまり「力場は XML、履歴と
オプティマイザ状態は pickle」という役割分担になっており、`ffinfo` を pickle から
上書きするのは設計と矛盾する（ユーザーが渡した `chkpoint.xml` が無視される）。
`docs/source/thermodyn_perturb.md` の再開手順も pkl と xml の両方を渡す前提で書かれている。

そこでループを削除し、意味のある 2 つを明示的に書いた。

```python
# the force field itself comes from initial_ffxml, which BaseTrainer
# already loaded, so only the history is taken from the checkpoint
trainer.losses = dump_dict["losses"]
trainer.epochs = dump_dict["epochs"]
```

挙動は変わらない（`losses` / `epochs` には同じ値が入り、`ffinfo` はもともと復元されて
いない）。唯一の差は未使用の `trainer.epoch` 属性が作られなくなること。
`DistanceTrainer` / `SumTrainer` が `trainer.ff.ffinfo` を復元しているのは、
そちらが XML 経由の復元経路を持たないためで、2 クラスの方針の違いは意図的。

## fit が steps + 1 回まわっていた（2026-08-21 修正）

```python
end_epoch = start_epoch + steps + 1
for i_epoch in range(start_epoch, end_epoch):
```

`fit(steps=10)` が 11 エポック回っていた。`+ 1` を外し、`fit(10)` を 2 回呼ぶのと
`fit(20)` を 1 回呼ぶのが同じになることを docstring に明記した。

**破壊的変更**: 1 回の `fit` あたりのエポック数が 1 減る。既存のテストは
`len(trainer.losses) > 1` のような回数非依存の検査だけだったので影響はなかった。
回帰テストを 2 件追加（`fit(steps)` がちょうど steps 回まわること、分割して呼んでも
合計が一致すること）。

## 実効サンプル数チェックが状態を位置で照合していた（2026-08-21 修正）

DMFF の `estimate_effective_sample(..., decompose=True)` は**状態名をキーにした辞書**を
返す。順序は `estimator.states` の並び順。

```python
for ii in range(len(self.sampling_params)):        # ii = レプリカ番号
    for i, (k, v) in enumerate(ieff.items()):      # i  = 辞書内の位置
        if v < self.neff[ii] and k != "Total" and ii == i:
            self.resample[i] = True
```

キーは名前で入っているのに、コードは名前を見ずに位置で照合していた。これが成り立つのは
`estimator.states` の並びがレプリカ番号と一致している間だけで、`_resample` は状態を
remove してから add し直すため、一度でも部分的な再サンプリングが起きると並びがずれる。

3 レプリカのうち 1 番だけ再サンプリングすると並びは `[0, 2, 1]` になる。この状態で
`sample_1` だけが枯渇したときの判定:

```
ieff: {'sample_0': 100, 'sample_2': 100, 'sample_1': 3, 'Total': 203}
  レプリカ 0: 旧=False 新=False
  レプリカ 1: 旧=False 新=True    <- 判定が違う
  レプリカ 2: 旧=True  新=False   <- 判定が違う
```

枯渇したレプリカ 1 が再サンプリングされず、健全なレプリカ 2 が無駄に再サンプリングされる。
例外は出ないので気づけない。

判定を `_needs_resample(ii, ieff)` に切り出し、`ieff.get(_state_name(ii))` と名前で
引くようにした。`k != "Total"` のガードは名前照合では不要になる。

### 併せて: _resample のインデックス取り違え

```python
removedstatename = [states[i].name for i, flag in enumerate(self.resample) if flag]
removedstateidx  = [i for i, flag in enumerate(self.resample) if flag]
for idx in removedstateidx:
    self.estimator.remove_state(removedstatename[idx])
```

`removedstatename` は**絞り込み後**のリスト（長さ = 再サンプリング対象数）なのに、`idx` は
**元のレプリカ番号**。`resample = [False, True]` だと長さ 1 のリストを `[1]` で引いて
`IndexError` になる。先頭から連続して対象になる場合しか通らなかった。

```
resample=[False, True] -> removedstatename=['sample_1'] removedstateidx=[1]
  旧: removedstatename[1] -> IndexError: list index out of range
  新: _resample_indices() -> [1]  (名前は sample_1 を直接生成)
```

対象の選択を `_resample_indices()` に切り出し、状態名は `_state_name(idx)` で直接
組み立てるようにした。状態名の生成をモジュール関数 `_state_name` に一本化したので、
`setup` / `_resample` / `_needs_resample` の 3 箇所が同じ規則を共有する。

回帰テストを 6 件追加（並び替え後の名前照合、Total を拾わないこと、未登録の状態で
落ちないこと、対象選択 2 通り、状態名の一元化）。

## sub に登録した after_update フックが無視されていた（2026-08-21 修正）

```python
self.trainer1._do_modify("after_update", self.trainer1.ffparams)   # 戻り値を捨てている
self.trainer2._do_modify("after_update", self.trainer2.ffparams)
self.ffparams = self._gather_from_subtrainers()
```

`_do_modify` は加工後の値を**返す**だけなので、代入しないと反映されない。親側は
`self.ffparams = self._do_modify(...)` と代入していたが、sub 側は捨てていた。

厄介なのは、フックの書き方によって症状が変わる点。実測した組み合わせ表（修正前）:

| 登録先 | hook | 書き方 | 効くか |
| --- | --- | --- | --- |
| sub | `after_grad` | pure | ○ |
| sub | `after_grad` | in-place | ○ |
| sub | **`after_update`** | **pure** | **×** |
| sub | `after_update` | in-place | ○ |
| 単体 fit | `after_update` | pure / in-place | ○ |
| 親 | `after_update` | pure / in-place | ○ |

- **pure**（引数を触らず新しい値を返す）は戻り値が唯一の出力なので、捨てられると消える
- **in-place**（引数の入れ子 dict を破壊的に書き換える）は戻り値と無関係に効いてしまう

つまり「単体 fit では動くのに SumTrainer に入れると効かない」という、切り分けにくい形で
現れる。しかもエラーも警告も出ない。`after_grad` は `_substep` で
`grads = trainer._do_modify(...)` と代入しているため、どちらの書き方でも効いていた。

修正は戻り値の代入。あわせて実行順（sub が自分の半分 → 連結 → 親が全体）をコメントにした。

```python
self.trainer1.ffparams = self.trainer1._do_modify(
    "after_update", self.trainer1.ffparams
)
self.trainer2.ffparams = self.trainer2._do_modify(
    "after_update", self.trainer2.ffparams
)
self.ffparams = self._gather_from_subtrainers()
```

修正後は 8 パターンすべてが効く。

### 契約の明文化

同じ間違いを誘発しないよう、`add_modifyfn` の docstring に契約を書いた。

> The hook takes the value and **returns** the modified one; the caller assigns
> that return value. It must not rely on modifying its argument in place, and
> it must not return None.

`_do_modify` も、フック未登録のときに None ではなく第 1 引数をそのまま返すようにした。
呼び出し側が常に戻り値を代入できるようになる（`after_grad` / `after_update` は
`__init__` で必ず登録されるので、この分岐は現状到達しない防御）。

回帰テストを 4 件追加（sub の pure な `after_update` が効くこと、親のフックが sub へ
配られること、sub → 親の順で呼ばれ親は 2 倍の長さのベクトルを見ること、未登録の
フック種別で値がそのまま返ること）。

## sub の optimizer が「適用されない更新」で進んでいた（2026-08-21 修正）

`_substep`（元は `get_loss_gradients` にインラインで 2 回書かれていた）の最後の行。

```python
_, trainer.opt_state = trainer.optimizer.update(grads, trainer.opt_state)
return loss, grads
```

`optax` の `update` は `(updates, new_state)` を返すが、ここでは `updates` を捨てて
`new_state` だけ保存していた。実際のパラメータ更新は親の optimizer が
連結・重み付け後の勾配から行うので、sub の optimizer は毎ステップ呼ばれて内部状態を
進めるだけで、その出力は一度も使われない。

リファクタ前のコードに経緯が残っていた。

```python
updates, self.trainer1.opt_state = self.trainer1.optimizer.update(...)
# self.trainer1.ffparams = optax.apply_updates(self.trainer1.ffparams, updates)
```

もともと sub ごとに独立更新する設計で、「連結して親でまとめて更新する」方式に変えた際に
apply の行だけコメントアウトし、`update` の呼び出しが残った。

### 何が問題だったか

Adam の `opt_state` はステップカウンタと 1 次・2 次モーメントを持つ。

1. **checkpoint に幻の状態が入る**。3 クラスの `write_checkpoint` はいずれも `opt_state`
   を保存し、`SumTrainer.write_checkpoint` は最後に両 sub のそれを呼ぶ。その sub を後から
   単体で `from_checkpoint` すると、「モーメンタムは持っているが、それに対応する
   パラメータ変化は別の optimizer が別の勾配で行った」という食い違った状態から始まる。
2. **勾配のスケールが親と違う**。sub が見るのは重み付け前の勾配、親が見るのは
   `weight` を掛けた勾配。`weight=[1.0, 5.0]` なら 2 次モーメントのスケールが 5 倍ずれる。
3. 捨てるためにクリッピングとモーメント更新を毎ステップ 2 回計算していた。

### 修正

`optimizer.update` の呼び出しを削除し、`after_grad` フックの結果をそのまま返す。

```python
return loss, trainer._do_modify("after_grad", grads)
```

「sub は勾配の供給元で、更新は親が行う」という役割分担を docstring に明記した。

```
3 step 後の count : sub(t1)=[0, 0]  sub(t2)=[0, 0]  親=[3, 3]     （修正前は sub も [3, 3]）
```

sub の `opt_state` は `setup()` で初期化されたまま動かなくなるので、checkpoint に入るのも
初期状態になり、「単体で再開したら最初から」という素直な意味になる。パラメータの更新経路は
親の optimizer だけなので、学習結果そのものは変わらない。

回帰テストを 2 件追加（親だけがステップを進めること、`optimizer.update` を外しても
`after_grad` フックは効くこと）。
