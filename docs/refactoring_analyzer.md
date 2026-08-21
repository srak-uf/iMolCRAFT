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

1. `calc_rdf_frame` は `mda.InterRDF` インスタンスを 1 つ作り、フレームごとに `run(frames=[i])`
   を呼んで `results.rdf` を使い回している。MDAnalysis が `run()` のたびに `results.rdf` を
   新しい配列に差し替える実装に依存しており、将来 in-place 更新に変わると全フレームが
   同じ配列を指してしまう。フレームごとにインスタンスを作るか、`np.array(g)` でコピーするのが安全。
2. `_drop_spurious_first_bin` は `g[0] > 1` という閾値で最初のビンを 0 にするが、
   1 という値の根拠がコードにもコメントにも無い。`dr` が大きいと正当なピークを潰しうる。
3. `calc_adf_frame` の `rcut12 /= 10.0` は引数を関数内で書き換えている。呼び出し側の値は
   変わらない（float なので）が、単位変換が「angstrom -> nm」であることが読み取りにくい。
   今回コメントを足したが、`rcut12_nm = rcut12 / 10.0` と別名にするほうが明確。
4. `calc_adf` は `calc_adf_frame` の結果をフレーム平均するが、フレームごとに
   `density=True` で規格化した後で平均している。フレーム間で三つ組の本数が違う場合、
   本数の少ないフレームが同じ重みで効く。全フレームの角度をまとめてヒストグラムにするのとは
   結果が異なる。意図的かどうか要確認。
5. `analyzer/__init__.py` の `from .analyzer import *` は `__all__` が無いため、
   `np` / `md` / `mda` / `MDAnalysis` まで `imolcraft.analyzer` 名前空間に露出している。
   `__all__` を定義すれば整理できるが、`imolcraft.analyzer.np` のような参照をしている
   ノートブックがあると壊れるため今回は触っていない。
6. `calc_density_frame` の `sum(u.atoms.masses)` は Python の逐次加算。`.sum()` にすると
   わずかに速く数値的にも安定だが、浮動小数点の加算順が変わり結果がビット単位で変わりうるため
   今回は据え置いた。
