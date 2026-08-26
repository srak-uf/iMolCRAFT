# analyzer リファクタリング記録

対象: `imolcraft/analyzer/analyzer.py`（363 行 -> 406 行、うち増分の大半は docstring 補完）

## 方針
crafter と同じ。ファイル構成は維持（import 経路を変えない）、挙動は完全維持（等価変換のみ）、
潜在バグは修正せず記録する。詳細は `docs/refactoring_crafter.md` を参照。

## 検証方法
- テスト環境: `~/miniconda3/envs/imc_cpu/bin/python`。analyzer 専用のテストは存在しない。
- 旧コード（git HEAD 版）を別モジュールとしてロードし、実トラジェクトリ
  （`tests/vs_supercell_bonds.pdb` + `tests/xtcfiles/sample_0.xtc`、12 原子 / 5 フレーム）で
  全 7 関数の戻り値を `np.array_equal` で突き合わせた。**66 項目すべてビット単位で一致**。
  - `calc_rdf` / `calc_rdf_frame`: 元素対 3 通り × `only_intermolecular` 2 通り ×
    フレームスライス 2 通り
  - `calc_adf_frame` / `calc_adf`: 元素三つ組 3 通り × カットオフ 2 通り × スライス 2 通り
  - `calc_density_frame` / `calc_density` / `calc_cellpar_frame`（target 7 通り）
  - 不正な `target` を渡したときの `ValueError` メッセージ文字列も一致
- `pytest tests/test_trainer tests/test_crafter -m "not qm"` は 22 passed / 7 failed / 1 error。
  失敗と error は packmol / antechamber / psi4 の未インストールによる環境要因でリファクタ前と同じ。
  `trainer/dmff_utils.py` が analyzer の 4 関数を使っているため、この結果が結合側の確認になる。

## 主な変更

| 変更 | 内容 |
| --- | --- |
| `_build_interrdf` 抽出 | `calc_rdf` と `calc_rdf_frame` に完全重複していた「原子選択 + InterRDF 構築」7 行を集約。`exclude_same` の有無で `mda.InterRDF(...)` の呼び出しが 2 つに分岐していたのを kwargs 辞書に統一 |
| `_drop_spurious_first_bin` 抽出 | 2 箇所にあった `if g[0] > 1: g[0] = 0.0` を関数化し、なぜ 0 にするのか（r~0 でシェル体積が消えて規格化が発散する）を docstring に記載 |
| `_join_on_center` 抽出 | `calc_adf_frame` の三重ループを、中心原子をキーにした辞書引きに置換。三つ組の生成順は従来どおり |
| `_element_pairs` 抽出 | `range(len(...))` の二重内包表記を要素の直接反復に |
| `arr_1_2` / `arr_2_3` の巻き上げ | `np.array(pairs_1_2)` をフレームごとに作り直していたのをループ外へ |
| `_ADF_BINS` 定数化 | `np.arange(0, 180 + 0.001, 1)` が `calc_adf_frame` と `calc_adf` に重複していた |
| `_AMU_TO_G` / `_ANG3_TO_CM3` 定数化 | `calc_density_frame` の `1.66053886e-24` と `1e-24` に名前を付けた |
| `CELLPAR_INDICES` 辞書化 | `calc_cellpar_frame` の 7 分岐 if-elif を辞書引きに。エラーメッセージも同じ辞書から生成するので、パラメータを増やしたときにメッセージが古くならない（文字列は従来と完全一致） |
| コメントアウト済みコードの削除 | `calc_adf_frame` の「泥臭いコード」と題された未使用の内包表記 5 行 |
| `rcut12_nm` / `rcut23_nm` へ改名 | `rcut12 /= 10.0` と引数を上書きしていたのを別名の変数に。angstrom -> nm の単位変換であることが名前から分かる |
| docstring の補完・修正 | `calc_rdf_frame` に欠けていた `only_intermolecular`、`calc_adf` に欠けていた `start` / `stop` / `step` を追記。`calc_cellpar_frame` の戻り値形状の記述が誤り（単一 target で `(n_frames, 1)` と書かれていたが実際は `(n_frames,)`）だったので実測に合わせて修正 |

### `_join_on_center` の高速化

旧実装は各フレームで「カットオフを通った 1-2 ペア」×「カットオフを通った 2-3 ペア」を
総当たりし、中心原子が一致するものを拾っていた。ペア数の積に比例するため、原子数が増えると
急速に重くなる。中心原子をキーにした `defaultdict` を一度作れば、走査はペア数の和 + 実際の
一致数に比例する。

ランダム入力での実測（出力が旧実装と完全一致することを確認済み）:

```
pairs=200x200   triplets=805 : old 0.007s -> new 0.000s
pairs=800x800   triplets=3107: old 0.111s -> new 0.001s
pairs=2000x2000 triplets=8058: old 0.692s -> new 0.014s
```

これがフレーム数だけ繰り返されるので、長いトラジェクトリの ADF ほど効く。

## 未修正の潜在バグ・気になる点

現時点で残っているものは無い。報告した 6 件はすべて修正済みか、仕様として確認済み。

## 修正済みバグ

### calc_rdf_frame が results.rdf の実装詳細に依存していた（2026-08-21 修正）

`calc_rdf_frame` は `mda.InterRDF` のインスタンスを 1 つだけ作り、フレームごとに
`run(frames=[i])` を呼んで `rdf.results.rdf` をリストに追加していた。

```python
rdf.run(frames=[i_frame])
g = rdf.results.rdf      # 参照を保持
rdf_list.append(g)
```

これは「MDAnalysis が `run()` のたびに `results.rdf` を新しい配列に割り当て直す」という
実装詳細に依存している。MDAnalysis 2.10.0 では実際に毎回別オブジェクトになるため
（`id()` が毎回変わることを確認）現時点で実害は無いが、将来 in-place 更新に変わると
リストの全要素が同じ配列を指し、全フレームが最後のフレームの値に潰れる。

修正は明示的なコピー。

```python
rdf_list.append(_drop_spurious_first_bin(np.array(rdf.results.rdf)))
```

`_drop_spurious_first_bin` はコピー側を書き換えるので `rdf.results.rdf` には触れなくなるが、
その値は次の `run()` で上書きされるだけなので戻り値は従来と同一。

検証: `InterRDF._conclude` を「毎回同じバッファに書き戻す」実装に差し替えて注入したところ、
旧実装は全フレームが同一値に潰れ、新実装は通常時と同じ結果を返した。通常の MDAnalysis 2.10.0
での 66 項目の等価性チェックも引き続き全一致。

### RDF の最初のビンのゼロ化が条件付きだった（2026-08-21 修正）

意図は「g[0] すなわち r = 0 の RDF はゼロであるべき」という物理的要請（ユーザー確認済み）。
しかし実装は `if g[0] > 1` という条件付きだった。

```python
if g[0] > 1:
    g[0] = 0.0
```

`dr` が小さければ、ビン 0 にカウントが 1 つでも入ると規格化（シェル体積 ~ dr^3 で割る）で
g が巨大になるため、この条件は「カウントが入ったか」の代用として実用上は機能する。
ただし意図をそのまま表してはおらず、`0 < g[0] <= 1` や NaN の場合に値が残る。

修正は条件の除去。

```python
g[0] = 0.0
```

ヘルパー名も `_drop_spurious_first_bin` から `_zero_first_bin` に変更した。旧名は
「異常値を捨てる」というヒューリスティックに読め、実際の意図とずれていた。

挙動差（ヘルパー単体で確認）:

| 入力 g[0] | 旧 | 新 |
| --- | --- | --- |
| 0.0 | 0.0 | 0.0 |
| 0.3 | 0.3 | **0.0** |
| 1.0 | 1.0 | **0.0** |
| 1.0000001 | 0.0 | 0.0 |
| 477.46 | 0.0 | 0.0 |
| NaN | NaN | **0.0** |

実データ（12 原子 5 フレームの C-H RDF、rmax=8.0）では `dr` を 0.01 から 8.0 まで振っても
旧実装と結果が完全に一致した。`0 < g[0] <= 1` になる状況が現れなかったため。
既定の `dr` = 0.01 A での結果は変わらない。

なお「`dr` が大きいとビン 0 が実ピークを飲み込む」という問題は本修正の対象外で、依然として残る。
実測では `dr` = 2.0 A でビン 0 が [0, 2.0) となり、C-H 結合ピーク（1.09 A）の 20 カウント
（g[0] = 477）がゼロ化される。既定の `dr` = 0.01 A では起きない。

## テストの新設

analyzer にはテストが 1 件も無かったので `tests/test_analyzer/test_analyzer.py` を新設し、
13 件を追加した。

- `_zero_first_bin`: 入力値によらず g[0] が 0 になること、他のビンを触らないこと、
  in-place であること（`np.nan` を含む 6 通りをパラメータ化）
- `calc_rdf` / `calc_rdf_frame`: 最初のビンが 0 であること、フレームごとの行が
  同じ配列を共有していないこと（results.rdf のエイリアシング修正の回帰テスト）
- `_join_on_center`: 中心原子を共有する三つ組の生成順、一致が無い場合
- `calc_density_frame` / `calc_density`: 形状と平均の整合
- `calc_cellpar_frame`: target ごとの形状、不正な target で ValueError

テストデータは既存の `tests/vs_supercell_bonds.pdb` + `tests/xtcfiles/sample_0.xtc`
（12 原子 / 5 フレーム）を使うので追加ファイルは不要。

## 仕様として確認済みの挙動

以下は「バグではないか」と報告したが、ユーザーに確認して意図どおりと判明したもの。
再び疑われないようコード中にコメントとして残した。

### RDF の最初のビンをゼロにすること

r = 0 の RDF は物理的にゼロであるべき、というのが根拠。`_zero_first_bin` の docstring に記載。
（条件付きだった実装は上記のとおり無条件に修正済み）

### calc_adf がフレームごとに規格化してから平均すること

`calc_adf_frame` が `density=True` でフレーム単位に規格化した分布を返し、`calc_adf` は
それを単純平均する。フレーム間で三つ組の本数が違っても各フレームが等しい重みで効く。
全フレームの角度をプールして 1 つのヒストグラムにするのとは結果が異なるが、
「フレームごとの分布の平均」が欲しい量なのでこれが正しい。`calc_adf` にコメントとして記載。

### __all__ が無く import したモジュールが漏れていた（2026-08-21 修正）

`analyzer/__init__.py` は `from .analyzer import *` だけで、`analyzer.py` に `__all__` が
無かったため、`imolcraft.analyzer` 名前空間に以下まで露出していた。

```
CELLPAR_INDICES, MDAnalysis, analyzer, calc_adf, calc_adf_frame, calc_cellpar_frame,
calc_density, calc_density_frame, calc_rdf, calc_rdf_frame, defaultdict, md, mda, np
```

`MDAnalysis` / `md` / `mda` / `np` / `defaultdict` は実装の都合で import しただけのもので、
API ではない。`imolcraft.analyzer.np` のような参照が書けてしまい、後で import を整理すると
無関係な箇所が壊れる。

`analyzer.py` に `__all__` を定義して公開 API 7 関数 + `CELLPAR_INDICES` に絞った。
リポジトリ内に漏れていた名前を参照している箇所は無いことを確認済み。
`imolcraft.analyzer.analyzer.np` のようなサブモジュール経由のアクセスは従来どおり有効。

回帰テスト `test_package_exports_only_public_api` を追加した。

### total_mass の逐次加算を .sum() に（2026-08-21 修正）

`calc_density_frame` の `sum(u.atoms.masses)` は numpy 配列を Python の逐次加算で畳んでいた。
`.sum()` は pairwise summation を使うため、速度と数値安定性の両方で優る。

```
n=      12: 逐次加算       0.7 us -> .sum()     0.5 us (   1.2x)
n=    1000: 逐次加算      36.5 us -> .sum()     1.5 us (  24.8x)
n=  100000: 逐次加算    3485.1 us -> .sum()    17.9 us ( 194.9x)
n= 1000000: 逐次加算   32837.2 us -> .sum()   140.0 us ( 234.6x)
```

加算順が変わるため密度は最終ビットが変わる。テストデータ（12 原子）での実測は相対差 2.09e-16
（約 1 ULP）で、`total_mass` が 80.08999999999997 から 80.08999999999999 になったことによる。
analyzer で唯一ビット単位の一致が崩れる箇所なので、等価性スクリプトはこの 2 関数だけ
相対許容差 1e-15 で比較している。他の 64 項目は引き続き完全一致。
