# crafter リファクタリング記録

対象: `imolcraft/crafter/` (asemol.py / ffxml.py / gaffil_generators.py / molinfo.py)

## 方針
- ファイル構成は維持（import 経路を変えない）。内部整理のみ。
- 挙動は完全維持（等価変換のみ）。潜在バグは修正せず本ドキュメントに記録。

## 検証方法
- テスト環境: `~/miniconda3/envs/imc_cpu/bin/python`（`imolcraft` が editable install 済み）。
  `imc_dev` / `.conda` には imolcraft が入っていないので使えない。
- `python -m pytest tests/test_crafter -m "not qm"` の結果はリファクタ前後で同一
  （11 passed / 7 failed）。7 failed は `packmol` / `antechamber` / `parmchk2` /
  `psi4` が未インストールなことによる環境要因で、リファクタとは無関係。
- 旧コード（git HEAD 版）を別モジュールとしてロードし、同一入力で出力を突き合わせる
  等価性スクリプトを作成して確認済み。
  - asemol: 結合リスト・分子分割・アンラップ座標・`get_ase_molecules` の全出力・
    PDB 出力バイト列・スーパーセル高速経路（5184 原子 / 3456 結合）まで完全一致。
  - molinfo: `from_yaml` → `prep(do_opt=False, do_charge=False)` → `_adjust_charges`
    → `get_smiles` の全 mol_info フィールドが一致。
  - gaffil_generators: antechamber をスタブして `run_antech` の出力 mol2 を比較し一致
    （FSA / PF6 両方）。

## 主な変更

### asemol.py
| 変更 | 内容 |
| --- | --- |
| `_calculate_bonds` のベクトル化 | 二重 for + `DataFrame.loc` 参照 → `np.triu_indices` + 閾値行列。5184 原子で 69.8s → 10.1s |
| `_bond_thresholds` 新設 | 元素記号 → 最大結合長の行列参照を一箇所に集約 |
| `get_molecules` の DFS を反復化 | 再帰 → スタック（訪問順は保存）。大規模系の RecursionError 回避 |
| `unwrap_molecules` | 各原子ごとに全結合を線形走査 → 隣接リストを一度だけ構築 |
| `_find_intercell_bonds_simple` | 1 ペアずつ `get_distances` → セル対ごとにバッチ化。`nx, ny, nz` が `networkx as nx` を隠していたため `n_x, n_y, n_z` に改名 |
| `_read_bond_definitions` / `_find_boundary_atoms` 抽出 | `__init__` と高速経路から前処理を分離 |
| `aseatoms2pdb` | 単原子残基/通常残基で完全重複していた 16 引数の format 呼び出しを 1 箇所に統合 |
| `is_same_molecule` / `reorder_atoms` | 共通の `_element_graph_matcher` に集約 |
| `pdb2packmol` | 3 箇所に散っていた質量集計を `_total_mass`、セル長計算を `_cubic_cell_length`、個数決定を `_nmols_from_density` に抽出。分子ごとの結合計算を `for _ in range(nmols[i])` の外へ巻き上げ |
| `merge_asemols` | 関数名と同名のローカル変数のシャドーイングを解消 |

### ffxml.py
- 入れ子関数 `_write_PF6` / `molecule2aseatoms` / `get_element_angles` を
  モジュールレベルの `_write_pf6_xml` / `_molecule2aseatoms` / `_get_element_angles` へ。
- `gafftemplate2xml` のループ本体を `_fill_charges_and_conformer` / `_write_pf6_xml` /
  `_write_ion_xml` に分割し、本体は分岐だけになった。
- PF6 の 180 度 / 90 度の角度キー書き換えが完全重複していたので `(offset, group)` のループに統合。
- `merge_xml` の「index を集めて `pop(i - n_del)`」を、走査しながら残す要素だけ集める形に変更。
- `check_vsite` 内の重複 import を削除。

### gaffil_generators.py
- 関数途中に散っていた import（`parmed` / `lxml` / `StringIO` / `inspect` / `shutil` /
  `openff.units`）をすべてモジュール先頭へ。
- 可変オブジェクトのデフォルト引数 `il_assign={...}` → `None` + `DEFAULT_IL_ASSIGN` 定数。
  （元コードも None 時に同じ辞書へフォールバックしていたため挙動は不変）
- FSA の S / N / O 判定の 3 分岐 → `_FSA_METADATA_KEYS` のループ。
- `params.atom_types` を走査して片方の分岐が `pass` だけだった死んだループを
  `self._gaff_atom_types_observed.update(...)` に簡約。
- `generate_residue_template` から `_load_openmm_parameters` / `_append_residue` を抽出。
- `atom` / `bond` を `etree.SubElement` の戻り値で上書きしていたシャドーイングを解消。

### molinfo.py
- `build()` と `get_ffxml()` に丸ごと重複していた「分子収集 + テンプレートジェネレータ生成
  + XML 出力」を `_molecules_with_mol2` / `_make_fftemplate_generator` / `_generate_ffxml`
  に集約。
- `build()` の結晶 / 液体の構造構築を `_build_crystal_pdb` / `_build_liquid_pdb` に分離、
  トポロジへの結合追加は `_add_bonds_to_topology` に共通化。
- `from_yaml` の cif / molecules で重複していたパス解決を `_resolve_path` に統一。
- `keys is None` のときに全キーを使うイディオムが 8 メソッドに散っていたので `_keys_or_all` に統一。
- `get_optstructure` を `_resolve_geoopt_params`（バックエンド別デフォルトのマージ）/
  `_make_geoopt_calculator`（計算機の生成とトラジェクトリ名）/ `_find_similar_conformer`
  （RMSD による計算スキップ判定）に分割。ループ変数 `j` のリーク前提だった箇所は
  ヘルパーの戻り値で明示的に受け渡す形に変更。
- `_assign_totalcharge` を `_assign_molecule_charge` / `_distribute_unassigned_charge` に分割。
  入れ子関数だった `_extract_ring` / `_Ncation` はモジュールレベルの
  `_extract_ring` / `_has_imidazolium_ring` / `_count_quaternary_nitrogen` へ。
- 単原子イオンの電荷判定 if-elif 連鎖を `MONATOMIC_ION_CHARGES` の辞書引きに置換。
- `get_partial_charges` の `psi4_flag` 分岐で ChargeCalculator / Psi4ChargeCalculator の
  呼び出しが完全重複していたのでクラスを変数化して 1 箇所に。
- `"amber/ions/ionsff99_tip3p.xml"` などのマジック文字列を `DEFAULT_ION_FFXML` /
  `YAML_SECTIONS` / `GEOOPT_DEFAULTS` の定数へ。
- `supercell_bonds.pdb` の書き出しでクローズされていなかったファイルハンドルを `with` に。

## 未修正の潜在バグ（挙動維持のため据え置き）

1. `ffxml.gafftemplate2xml`: `ion_ffxml=None`（デフォルト）のまま単原子イオンを渡すと
   `os.path.join(..., None)` で TypeError。`Crafter` 経由では常に値が渡るので顕在化しない。
   なお同じ行の `ion_ffxml` 再代入自体は、2 回目以降が絶対パスの join になり同じ値を返すため
   実害はない。
2. `ffxml._write_ion_xml`: ion ライブラリに該当元素がないと `target_ptype` が未束縛のまま
   参照され UnboundLocalError。
3. `asemol.asemol_wrapper.__init__`: `bond_def_file` を明示指定すると `self.bond_def_file` が
   設定されないまま参照され AttributeError。既定値（None）でしか動かない。
4. `molinfo.get_optstructure`: `do_calc=False` だと `aseatoms_geoopt` が None のままなので
   末尾の `get_potential_energy()` で AttributeError。
5. `molinfo.from_crystal(assign_totalcharge=False)`: 警告 print がループ変数 `i` に依存。
   `molecule_list` が空だと NameError。
6. `asemol._find_intercell_bonds_simple`: 近傍セル index を `% n_x` で巻き戻すため、
   repeat が 1 の方向では自分自身のセルとの結合を判定する。
7. `asemol.get_ase_molecules`: 全 ref × 全分子で同型判定を試すため O(N^2) の isomorphism 呼び出し。
   分子数が多い系ではここが支配的になる。

## 修正済みバグ

### MOLINFO_KEYS のキー名不整合（2026-08-21 修正）

`MOLINFO_KEYS` は「1 分子分の情報辞書にどのキーがどの型の初期値で存在するか」を宣言する
スキーマだが、宣言と書き込み側のキー名がずれていた。

- 宣言は `"natoms"` / `"nmols"`（小文字）。書き込むのは
  `append_fromAtomsList` の `"Natoms"` / `"Nmols"`（大文字始まり）。
- `"symbol"` は `_assign_molecule_charge` が書き込むのに宣言がない。

Python の辞書は代入時に未知のキーを黙って追加するため、`_initialize_molinfo` が用意した
`natoms` / `nmols` は常に `None` のまま残り、実データは別キーに入る。実行時の実測では
宣言 15 個に対して辞書は 18 キーに膨れていた。`MOLINFO_KEYS` を見て `mol_info[key]["natoms"]`
と書いたコードは `KeyError` にならず `None` を返すため、エラーが出るのは値を使う地点まで
遅延する。

修正内容（案 A）:

- `MOLINFO_KEYS` の `"natoms"` / `"nmols"` を `"Natoms"` / `"Nmols"` に改名。
- `"symbol": type(None)` を追加。
- キー名を書き込み側と一致させる旨のコメントを付記。
- `_initialize_molinfo` の `v() if callable(v) else v` から、到達しない `else v` 側を除去して
  `{k: factory() for k, factory in MOLINFO_KEYS.items()}` に簡約。`MOLINFO_KEYS` の値は
  すべて型オブジェクト（＝常に callable）なので `else` 側は一度も実行されていなかった。
  併せて「値は 0 引数で呼べる callable であること」をコメントに明記した。

これで宣言と実際のキーが 16 個で完全一致する。小文字キーは本リポジトリ内のどこからも
参照されていなかったため（`structure["nmols"]` は YAML セクションのキーで別物）、削除による
影響はない。副次的な効果として、`_assign_totalcharge` を通していない `mol_info` でも
`info["symbol"]` が `KeyError` ではなく `None` を返すようになる。

注意: `load_crafter` で読み込む既存の pkl には旧キー構成がそのまま入っている。旧 pkl 側の
`natoms` / `nmols` は誰も読まないので実害はないが、pkl 由来の `mol_info` に対しては
`MOLINFO_KEYS` との一致を前提にできない。

テスト結果はリファクタ前と変わらず 11 passed / 7 failed。

### delvsite_pdb の空行での IndexError（2026-08-21 修正）

```python
lines = [line for line in lines if not line.split()[-1] == "EP"]
```

空行や空白のみの行では `line.split()` が `[]` を返すため `[-1]` で IndexError。
仮想サイトの判定を `_is_vsite_record` に切り出し、フィールドが 1 つも無い行は
仮想サイトではないと明示的に扱うようにした。判定基準（最後のフィールドが `EP`）は変更していない
ので、OpenMM が書いた PDB に対する結果は従来と同一。

`EP` は OpenMM の `PDBFile.writeFile` が `element is None` の粒子の元素欄に書く既定の識別子
（引数 `extraParticleIdentifier`）。この由来をモジュール定数 `_EXTRA_PARTICLE_IDENTIFIER` と
コメントで残した。

回帰テストを `tests/test_crafter/test_ffxml.py` に 2 件追加:

- `test_delvsite_pdb_removes_only_vsites`: EP レコードだけが消え、REMARK / MODEL / TER / END は残る。
- `test_delvsite_pdb_keeps_blank_lines`: 空行と空白のみの行があっても例外にならず、それらは保持される。
  旧実装ではこの入力で IndexError になることを確認済み。

### assert によるバリデーション（2026-08-21 修正）

`assert` 文は `python -O` / `PYTHONOPTIMIZE=1` で丸ごと除去されるため、入力検証には使えない。
`assert False, "..."` の形だと、最適化実行時にはチェックが消えて不正な入力がそのまま
後段に流れる。crafter 内の 6 箇所をすべて `raise ValueError` に置き換えた。

| 箇所 | 検出する条件 |
| --- | --- |
| `asemol.cast_molecules` | 2 つの分子グラフが非同型 |
| `molinfo._parser_yaml` | YAML に未知のトップレベルキー |
| `molinfo.append_fromAtomsList` | `atomslist` と `networkX` の長さ不一致 |
| `molinfo._make_fftemplate_generator` | 未知の力場タイプ |
| `molinfo._resolve_geoopt_params` | 未知の QM ソフト名 |
| `molinfo.get_partial_charges` | 不正な `charge_type` |

いずれも「呼び出し側が渡した値が不正」という条件なので `ValueError` を選んだ。
同じ性質のチェックである `asemol.pdb2packmol` の `fixed_property` / `priority_property` 検証と
`asemol.reorder_atoms` の非同型チェックは元から `ValueError` を送出しており、それに揃う。

合わせてエラーメッセージに実際の値を含めた（未知キーの一覧、長さの実数値、渡された
fftype / software / charge_type）。`charge_type` の許容値はマジックリストだったので
モジュール定数 `VALID_CHARGE_TYPES` に切り出した。

影響確認:

- `AssertionError` を捕捉しているコードはリポジトリ内に存在しない（`imolcraft/trainer/trainer.py`
  の `raise AssertionError` は別箇所で、捕捉側ではない）。
- テストに `pytest.raises` は 1 件も無く、例外型に依存したテストは無い。
- `asemol.get_ase_molecules` の `except Exception` は AssertionError / ValueError の
  どちらも同じく捕捉するため挙動不変。
- `python -O` で 6 箇所すべてが `ValueError` を送出することを実測で確認。
  旧実装は `-O` 下で不正値をそのまま通していた。

## 追記: run_antech の原子順の前提（2026-08-21、io リファクタ中に判明）

`GAFFilTemplateGenerator.run_antech` は `molecule.atoms`（OpenFF の原子順）から得た FSA の
metadata index を、mol2 ファイルの**行番号**としてそのまま使って GAFF 型を上書きする。

```python
for i, atom in enumerate(molecule.atoms):
    for meta_key, elem_key in _FSA_METADATA_KEYS:
        if atom.metadata.get(meta_key) is True:
            gaff_atoms[i][5] = fsa_types[elem_key]
```

つまり「OpenFF 分子の原子順 == mol2 ファイルの原子順」を暗黙に仮定している。通常の Crafter の
パイプラインでは mol2 を同じ OpenFF 分子から書き出すので成立するが、成立しない組み合わせでは
エラーにならず別の原子へ型が付く。`tests/data/fsa_resp.mol2.gaff` はまさにその状態で、
1 番目の原子（名前 `S`）に酸素の型 `o` が、2 番目（名前 `F`）に硫黄の型 `s6` が付いている。
これは SMILES から作った分子と無関係な RESP mol2 を組み合わせる
`tests/test_crafter/test_gaffilgenerator.py` の産物。

修正するなら、index ではなく原子名か元素で対応付けるか、少なくとも
`len(molecule.atoms) == len(gaff_atoms)` と元素の一致を検証して食い違いを検出すべき。
詳細は `docs/refactoring_io.md` の「補足」節を参照。
