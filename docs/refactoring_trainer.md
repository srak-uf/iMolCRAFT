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

1. `base.SumTrainer.write_checkpoint` は `dump_dict` を組み立てるが **`pickle.dump` を
   呼んでいない**。`with open(...) as f:` の中で辞書を作るだけで終わっており、
   `train_state_*.pkl` は空ファイルになる。他の 2 クラスの同名メソッドは正しく
   `pickle.dump` している。
2. `loss.mse_energy` の `weight_scheme="nonboltzmann"` は `IMPLEMENTED_WEIGHT_SCHEMES`
   に載っているが実装が `pass` のみで、`weight` が未定義のまま参照され
   `UnboundLocalError` になる。テスト側も `# , "nonboltzmann"` とコメントアウトされており、
   既知の未実装。
3. `base.SumTrainer.training_step` は sub-trainer の `after_update` フックの**戻り値を
   捨てている**。`self.trainer1._do_modify("after_update", ...)` の結果を代入していないため、
   sub-trainer にフックを登録しても効果がない。親側の同名フックは正しく代入している。
4. `base.SumTrainer.get_loss_gradients` は sub-trainer の `optimizer.update` を呼んで
   opt_state だけ進め、`updates` を捨てている。パラメータ更新は親側で行うので結果は
   正しいが、sub-trainer の opt_state が「適用されていない更新」の分だけ進む。
   モーメンタムを持つ optimizer では意味が変わりうる。
5. `dmff_utils._get_charges_types` は参照 0 件。
6. `base.BaseTrainer.fit` は `end_epoch = start_epoch + steps + 1` としており、
   `steps` に指定した回数より 1 回多く回る。
7. `trainer.ThermodynamicTrainer.after_step` の実効サンプル数チェックは
   `if v < self.neff[ii] and k != "Total" and ii == i` と、外側ループの `ii` と
   内側 `enumerate` の `i` を突き合わせている。状態数と分解項の数が一致する前提に
   見えるが、その保証はコードから読み取れない。
8. `trainer.ThermodynamicTrainer.from_checkpoint` の `attr_lists` に `"ff_info"` とあるが、
   保存側のキーは `"ffinfo"`。この setattr は一度も一致しない。
