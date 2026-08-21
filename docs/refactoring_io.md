# io リファクタリング記録

対象: `imolcraft/io/`（mol2.py / rdkit.py / exporter_lmp.py / exporter_gmx.py / _exporter.py、計 647 行）

## 方針
crafter・analyzer と同じ。ファイル構成は維持（import 経路を変えない）、挙動は完全維持
（等価変換のみ）、潜在バグは修正せず記録する。

## 検証方法
- テスト環境: `~/miniconda3/envs/imc_cpu/bin/python`。io 専用のテストは存在しなかった。
- 旧コード（git HEAD 版）を別モジュールとしてロードし、同一入力で突き合わせた。
  - **mol2 / rdkit: 31 項目一致**
    - `read_mol2` / `write_mol2`(dict) / `mol2_to_aseatoms` を実 mol2 3 ファイルで
    - `write_mol2`(Molecule) をエタンと FSA で（出力バイト列一致）
    - `atoms2rdkit` を FSA / PF6 / CH4 / 単原子イオン 4 種 × `il_assign` 2 通りで。
      SMILES・形式電荷・元素・結合次数のすべてを比較
    - `_il_assign` を FSA / PF6 / ClO4 / 該当なし 3 種で。戻り値だけでなく
      **入力分子への副作用**（形式電荷と結合次数の書き換え）も比較
  - **exporter: 7 項目一致（すべてバイト単位）**
    - `exporter_gmx` の .top / .gro、`exporter_lmp` の .data
      （parmed が .top ヘッダに出力ファイル名を埋め込むため、旧新とも同じ basename で
      別ディレクトリに書いて比較）
    - `to_lammps_non_rectangular` を box=None / 直方体 / 三斜晶の 3 分岐すべてで。
      三斜晶は Interchange の box を差し替えて到達させた
    - `file_path` に str と Path を渡した場合
- `pytest tests -m "not qm"` は 38 passed / 10 failed / 1 error。失敗と error は
  packmol / antechamber / parmchk2 / psi4 の未インストールによる環境要因で、
  リファクタ前と同じ顔ぶれ（crafter 7 件 + calculator 3 件 + trainer 1 件）。

## 主な変更

### mol2.py
| 変更 | 内容 |
| --- | --- |
| `write_mol2` の分岐整理 | `dict_flag` / `offmol_flag` の 2 つのフラグで型を持ち回していたのをやめ、書き出し関数を選ぶだけにした。到達不能だった最後の `else: raise` も除去 |
| `_write_mol2_dict` / `_write_mol2_molecule` 抽出 | 40 行あった `with open` ブロックの中身を用途別に分離 |
| `read_mol2` | `for i in range(len(lines))` + `lines[i]` を行の直接反復に。`readlines()` で全行をメモリに載せる必要もなくなった |
| `mol2_to_aseatoms` | docstring を追加し、`read_mol2` の結果を 2 回引いていたのを 1 回に |

### rdkit.py
| 変更 | 内容 |
| --- | --- |
| 入れ子関数の平坦化 | `_il_assign` の中に定義されていた `assign_pf6` / `assign_clo4` / `assign_fsalike` をモジュールレベルの `_assign_pf6` / `_assign_clo4` / `_assign_fsalike` へ。1 関数 194 行が最大 45 行の関数群になった |
| `_IL_ASSIGNERS` テーブル化 | 「呼ぶ → None でなければ print して return」の 3 回の繰り返しをループに |
| ガード節化 | 各 `assign_*` は本体全体が `if nc < 0:` に包まれ、該当しない場合は暗黙の `return None` に落ちていた。`if nc >= 0: return None` の早期 return に |
| `_set_formal_charge` / `_set_bond_type` 抽出 | 「3D 分子と 2D 分子の両方に同じ変更を適用する」処理が 6 箇所に散っていた |
| `_bond_end_with_symbol` 抽出 | `GetEndAtom()` / `GetBeginAtom()` を調べて相手側の index を取る処理の 4 重複を集約 |
| `_find_fsalike_n_s` 抽出 | FSA の N-S 骨格検出を分離 |
| 死んだコードの削除 | `assign_clo4` の `Cl_atom_list` / `Cl_nbond_list` / `Cl_highvalence_list` は計算されるだけで一度も使われていなかった |
| docstring の修正 | `assign_clo4` の説明が "Check if the molecule is PF6" というコピペ誤りだった |
| `atoms2rdkit` の分割 | `_molecule_from_positions`（座標から結合なしの RWMol を作る）、`_determine_bonds`（結合次数の推定とフォールバック）、`_monatomic_smiles`（単原子イオンの SMILES）に分離。3 分岐の if-elif-else だった SMILES 生成も 1 関数に |

### exporter_lmp.py

**このファイルへの変更は取り消され、HEAD の状態に戻っている。** ユーザーが後ほど大規模な
変更を予定しているため、現時点では触らない方針。以下は一度実施したが差し戻された内容の記録。

| 変更（差し戻し済み） | 内容 |
| --- | --- |
| `_LAMMPS_SECTIONS` テーブル化 | Bonds / Angles / ProperTorsions / ImproperTorsions について「件数の算出」「見出しの件数行」「型数の行」「係数セクション」「本体セクション」の 5 箇所すべてに同じ 4 分岐が書かれていた（計 20 個の分岐） |
| `_count_entries` / `_box_bounds` / `_write_masses` / `_unique_molecules` 抽出 | |
| `path` の未定義リスク解消 | `if isinstance(file_path, str)` / `if isinstance(file_path, Path)` の 2 段判定で、どちらにも該当しない型では `path` が未定義のまま使われる |
| 変数名 | `exporter_lmp` 内の `a = openmm.from_openmm(...)` が、`to_lammps_non_rectangular` 内で格子定数として使われる `a` と紛らわしい |

テスト（`tests/test_io/test_exporter.py`）は現行の HEAD 版に合わせてあり、公開関数
`exporter` / `exporter_lmp` / `to_lammps_non_rectangular` しか使っていない。ただし
Interchange を取り出すために `exporter_lmp_module.to_lammps_non_rectangular` を
monkeypatch しているので、大規模変更でこの構造が変わればテストの fixture も追随が必要。

### exporter_gmx.py / _exporter.py
- `open(system).read()` がファイルを閉じていなかったので `with` に（両 exporter）。
- `_exporter.exporter` の if/elif を `_EXPORTERS` 辞書のディスパッチに。
- 引数名 `format` / `input` は組み込み関数を隠していたので改名した（下記）。

## テストの新設

io にはテストが 1 件も無かったので `tests/test_io/` を新設し、31 件を追加した。

- `test_mol2.py`（10 件）: セクションの読み取り、dict のラウンドトリップ、
  不正な型で `ValueError`（かつ空ファイルを作らないこと）、`mol2_to_aseatoms` の元素と座標
- `test_rdkit.py`（13 件）: PF6 / ClO4 / FSA の検出、該当しない分子で None、
  該当しない分子の形式電荷を書き換えないこと、`_monatomic_smiles`、単原子イオンの変換
- `test_exporter.py`（8 件）: `_box_bounds` の 3 分岐、`_count_entries`、未知フォーマットで
  `ValueError`、gmx / lmp の実出力、LAMMPS data の見出しの並び

既存のテストデータ（`tests/data/*.mol2`、`tests/supercell_bonds.pdb`、`tests/system.xml`）で
足りるので追加ファイルは不要。

## 未修正の潜在バグ・気になる点

1. `exporter_lmp.to_lammps_non_rectangular` は box が無いとき警告なしに 100x100x100 A の
   セルを書き出す。密度が実際と違う LAMMPS data が黙って生成される。

## 修正済みバグ

### mol2_to_aseatoms が原子名をそのまま元素記号として使っていた（2026-08-21 修正）

```python
symbols = [atom[1] for atom in mol2_dict["@<TRIPOS>ATOM"]]
```

mol2 の ATOM レコードの index 1 は**原子名**で、元素記号そのものとは限らない。同じ元素が
複数ある分子では antechamber が `O`, `O1`, `O2`, `S`, `S1` のように連番を付けるため、
`Atoms(symbols=...)` が `KeyError: 'O1'` で落ちる（`tests/data/fsa_resp.mol2.gaff` で再現）。

修正は「名前から元素を取り出す」形にした。`_element_from_atom_name` が数字を除いたうえで
最長の元素記号の接頭辞（2 文字 → 1 文字の順）を返す。解決できない名前のときだけ
`_element_from_atom_type` で原子タイプ列にフォールバックする（`_element_of`）。

#### 名前列と型列のどちらを主にするか

一度は原子タイプ列（index 5）を主にする実装にしたが、実データを調べて名前列を主に切り替えた。

| 観点 | 名前列 | 型列 |
| --- | --- | --- |
| 元素との関係 | 元素記号が先頭に来る規約 | 力場の原子タイプ。元素とは間接的 |
| 体系の混在 | 単一 | SYBYL（`C.3`）/ GAFF（`s6`）/ 素の元素記号 が混在 |
| 対応表の要否 | 不要 | GAFF の `cl`/`br`/`si`（先頭 1 文字も元素）と `ca`/`na`/`no`（先頭 1 文字が正解）を区別する手書きの表が必要 |
| 型が壊れている場合 | 影響なし | 黙って誤った元素を返す |
| 弱点 | PDB 由来の名前（`CA` → Ca、`NE2` → Ne）を誤読する | 未知の力場タイプで `ValueError` |

リポジトリ内の全 mol2 に現れる原子名は 22 種
（`C C1 C2 C3 F F1 H H1 H2 H3 H4 H5 Li N N1 O O1 O2 O3 P S S1`）で、すべて「元素記号 + 連番」。
本物の antechamber 出力（`tests/test_crafter/ANTECHAMBER_AC.AC`）でも
`S→s6  F→f  O→o  O1→o  N→ne  S1→sy` と名前側は常に元素起点になっている。
PDB 由来の名前はこのパイプラインの mol2（`write_mol2` か antechamber の出力）には現れない。

決め手は 3 つ目の「型が壊れている場合」。`fsa_resp.mol2.gaff` は型列が
`o s6 o o n s6 o o o` と壊れており（下の「補足」を参照）、型方式では
`O S O O N S O O O` という誤った元素を静かに返す。名前列 `S F O O1 N S1 F1 O2 O3` からは
正しい `S F O O N S F O O` が得られる。

なお `ase.data.chemical_symbols` は先頭にダミーの `"X"` を持つため、`XX` のような名前が
誤って元素 `X` として解決されてしまう。両方の解決関数で `"X"` を除外している。

挙動の変化:

| ファイル | 名前列 | 型列 | 旧 | 新 |
| --- | --- | --- | --- | --- |
| `fsa_resp.mol2` | S F O O N S F O O | 同左 | 同左 | 変化なし |
| `pf6_resp.mol2` | P F F F F F F | 同左 | 同左 | 変化なし |
| `fsa_resp.mol2.gaff` | S F O **O1** N **S1** **F1** O2 O3 | o s6 o o n s6 o o o | `KeyError: 'O1'` | S F O O N S F O O |

回帰テストを `tests/test_io/test_mol2.py` に追加した（`_element_from_atom_name` 12 件、
`_element_from_atom_type` 22 件、`_element_of` の優先順位とフォールバック、
GAFF 出力の読み取り）。

### 補足: fsa_resp.mol2.gaff の型列が原子名と食い違っている件

上の表のとおり、このフィクスチャは 1 番目の原子が名前 `S` なのに型 `o` になっている。
調査したところ `crafter.GAFFilTemplateGenerator.run_antech` の挙動によるもので、
バグではなく前提条件の問題だった。

`run_antech` は `molecule.atoms`（OpenFF の原子順）を列挙して得た FSA の metadata index を、
mol2 ファイルの**行番号**として使って型を上書きする。両者の順序が一致していれば正しいが、
このフィクスチャは SMILES から作った分子と、無関係な RESP mol2 を組み合わせた
`tests/test_crafter/test_gaffilgenerator.py` の産物なので順序が違う。

```
OpenFF(SMILES)順の元素: ['O', 'S', 'O', 'F', 'N', 'S', 'O', 'O', 'F']
mol2 ファイル順の元素 : ['S', 'F', 'O', 'O', 'N', 'S', 'F', 'O', 'O']
FSA metadata index    : {'FSA_N': [4], 'FSA_S': [1, 5], 'FSA_O': [0, 2, 6, 7]}
index を行番号として適用: ['o', 's6', 'o', 'o', 'n', 's6', 'o', 'o', 'o']  <- .gaff の型列と一致
```

通常の Crafter のパイプラインでは mol2 を同じ OpenFF 分子から書き出すので順序は一致し、
問題は起きない。ただしこの前提はコードに書かれていないので、`run_antech` を単体で使うと
静かに別の原子へ型が付く。crafter 側の記録に残すべき事項。

### atoms2rdkit が numpy 整数の netcharge を受け付けなかった（2026-08-21 修正）

```python
assert isinstance(nc, int), f"nc must be an integer, but got {nc}"
```

問題が 2 つあった。

1. `numpy` の整数（`np.int64` など）は `int` のサブクラスではないので、この assert で弾かれる。
   netcharge を numpy 配列から取り出して渡すと落ちる。
2. `assert` は `python -O` で除去されるため、検証としては機能しない。

単に `isinstance(nc, (int, np.integer))` に広げるだけでは不十分で、RDKit の C++ バインディングが
numpy 整数を受け付けないことを実測で確認した。

```
DetermineConnectivity(charge=int64) -> ArgumentError: Python argument types in ...
DetermineConnectivity(charge=int32) -> ArgumentError: Python argument types in ...
DetermineConnectivity(charge=int)   -> ok
```

そこで受理と同時に Python の `int` へ正規化する形にした。

```python
if not isinstance(nc, (int, np.integer)):
    raise TypeError(f"nc must be an integer, but got {nc!r}")
# RDKit's bindings reject numpy integers, so normalise once here rather
# than at every call site downstream
nc = int(nc)
```

例外を `TypeError` にしたのは型の問題だからで、`python -O` でも消えないことを実測で確認した。
正規化により `_il_assign(mol, int(nc), mol2d)` の重複した `int()` も不要になった。

受理する値と拒否する値:

| 入力 | 結果 |
| --- | --- |
| `1` / `np.int64(1)` / `np.int32(1)` / `np.int8(1)` / `np.uint8(1)` | 受理（形式電荷 +1） |
| `1.0` / `np.float64(1.0)` / `"1"` / `None` | `TypeError: nc must be an integer, but got ...` |

浮動小数点を拒否する挙動は元のままにした（`1.0` を 1 と解釈すると、丸め誤差を含む値が
黙って通ってしまう）。回帰テストを 10 件追加した（numpy 整数 5 種、多原子分子で
Python int と同じ SMILES になること、拒否する型 4 種）。

### read_mol2 が壊れた mol2 で NameError になっていた（2026-08-21 修正）

```python
for line in f:
    if line.startswith("@<TRIPOS>"):
        key_name = line.strip()
        ...
    elif line.strip() != "":
        mol2_dict[key_name].append(...)   # key_name が未定義になりうる
```

最初の見出しより前に中身のある行があると、`key_name` が一度も代入されないまま参照されて
`NameError: name 'key_name' is not defined` になる。原因の行も、そもそも mol2 として
おかしいという事実も分からない。

行番号とファイル名を添えた `ValueError` を送出するようにした。

```
/tmp/.../broken.mol2:1: content before the first @<TRIPOS> section header:
'# Created by SomeTool'. Is this a mol2 file?
```

回帰テストを 3 件追加した（見出し前に中身がある場合、先頭の空行は許容されること、空ファイル）。

注釈行の扱いは下記のとおり別途対応した。

### read_mol2 が注釈行をデータとして取り込んでいた（2026-08-21 修正）

mol2 の仕様では `#` で始まる注釈行をどこにでも書ける。しかし `read_mol2` は
「見出しでも空行でもない行」をすべてデータ行として扱っていたため、

- セクション内の注釈は、そのセクションのデータ行として辞書に入る
- 最初の見出しより前の注釈は、上記の `ValueError` で弾かれる

という状態だった。`# Created by SomeTool` のようなヘッダを持つ正当な mol2 が読めない。

空行と同じ扱いで読み飛ばすようにした。

```python
stripped = line.strip()
# blank lines and "#" comments carry no data and may appear anywhere
if stripped == "" or stripped.startswith("#"):
    continue
```

インデントされた注釈も落とすため、`line` ではなく `strip()` した結果で判定している。
一方セクション見出しの判定は従来どおり行頭（`line.startswith("@<TRIPOS>")`）のままで、
インデントされた見出しはデータ行として扱われる挙動を変えていない。

注釈も空行も辞書には残らないので、読んで書き戻すと消える。この点を `read_mol2` の
docstring に明記した。回帰テストを 1 件追加（見出し前・セクション内・インデント付きの
3 種類の注釈を含む mol2 を読む）。

### 引数名が組み込み関数を隠していた（2026-08-21 修正）

`write_mol2(filename, input)` と `exporter(pdb, system, filename, format)` の引数名は、
組み込み関数の `input` と `format` を関数本体の中で覆い隠していた。

```python
def exporter_demo(format):
    format(0.5, ".3f")     # TypeError: 'str' object is not callable
```

影響はその関数本体の中だけなので現状のコードは動く。問題になるのは後から手を入れるときで、
たとえば `exporter` に進捗表示を足そうとして `format(size / 1e6, ".1f")` と書くと
実行時に落ちる。しかもエラーが `'str' object is not callable` なので、原因が引数名にあると
気づきにくい。`flake8-builtins` の `A002` が警告する対象。

| 変更前 | 変更後 |
| --- | --- |
| `write_mol2(filename, input)` | `write_mol2(filename, data)` |
| `exporter(pdb, system, filename, format)` | `exporter(pdb, system, filename, fmt)` |

`write_mol2` の呼び出しはリポジトリ内すべて位置引数なので影響なし。
`exporter` はノートブック 4 ファイル 7 箇所が `format="gmx"` とキーワードで呼んでいたので、
`fmt=` に更新した（差分は該当行のみ）。

- `examples/exporter/exporter.ipynb`（2 箇所）
- `examples/crafter/liquid/liquid.ipynb`（2 箇所）
- `examples/crafter/crystal/crystal.ipynb`（2 箇所）
- `examples/crafter/pf6/check_pf6.ipynb`（1 箇所）

**破壊的変更**: リポジトリ外のスクリプトやノートブックが `exporter(..., format="gmx")` と
書いている場合、`TypeError: exporter() got an unexpected keyword argument 'format'` になる。
互換が必要なら `format=None` を残して非推奨警告を出す移行措置を入れる余地がある。

回帰テストを 2 件追加（`exporter(..., fmt=...)` と `write_mol2(..., data=...)` が
キーワード引数で呼べること）。

### _assign_pf6 / _assign_clo4 が判定前に分子を書き換えていた（2026-08-21 修正）

両関数は「該当するイオンか」を確定する前に形式電荷を書き換え、判定が外れて `None` を
返してもその書き換えを戻さなかった。

```python
if not any(atom.GetSymbol() == "P" for atom in atoms):   # ゆるい前提チェック
    return None
for i, atom in enumerate(atoms):
    if symbol == "F":
        _set_formal_charge(atom, atoms2d, i, 0)          # ここで書き換え
    elif symbol == "P":
        _set_formal_charge(atom, atoms2d, i, -1)
if len(pf6like_Findex) == 6 and len(pf6like_Pindex) == 1: # 本判定はこの後
    return {...}
return None                                              # 書き換えは残ったまま
```

「陰イオンで P を 1 個でも含む」だけで書き換えに進んでしまう。`_assign_clo4` はさらに広く、
「4 配位の Cl が 1 個でもある」だけで**分子内の全 O** を -1、**全 Cl** を +3 にしていた。
なお `_assign_fsalike` は判定を先に済ませてから書き換えるので、この問題は無かった。

#### 被害

P の形式電荷を -1 にすると RDKit が原子価を合わせるため暗黙の水素を 1 個追加する。
結果として分子式と正味電荷が変わる。

| 分子 | before | after |
| --- | --- | --- |
| ジフルオロリン酸イオン `[O-]P(=O)(F)F` | `F2O2P⁻`（H 0、電荷 -1） | `HF2O2P²⁻`（H 1、電荷 **-2**） |
| リン酸二水素イオン `[O-]P(=O)(O)O` | `H2O4P⁻`（H 2、電荷 -1） | `H3O4P²⁻`（H 3、電荷 **-2**） |

`atoms2rdkit` は書き換えられた `mol` をそのまま返し、`Crafter.get_rdkitmol` →
`get_smiles` / `get_sdf` → `get_molecule_off` と伝播するため、誤った水素と電荷を持つ
OpenFF 分子が電荷計算・力場生成へ流れる。LiPO2F2 のようなリン酸系アニオンを扱うと踏む。

さらに `_assign_clo4` は誤検出も起こしていた。過塩素酸メチル `CO[Cl+3]([O-])([O-])[O-]` は
分子中の O が 4 つあるため ClO4 と判定され、Cl と結合していないエステル酸素まで -1 にされた。

#### 修正

検出（読み取りのみ）と適用（書き換え）を分け、共通の検出処理を
`_central_ion_indices(atoms, center_symbol, ligand_symbol, n_ligands)` に切り出した。

判定条件も見直した。中心原子がちょうど 1 つで、その隣接原子がちょうど `n_ligands` 個、
かつ**すべてが終端（degree 1）の配位子元素**であることを要求する。終端条件が、裸のイオンと
エステルなどの置換体を区別する鍵になる。

**訂正**: 直前の説明で「O の数え方を『分子中の全 O』から『その Cl に結合している O』に
直せば過塩素酸メチルの誤検出も消える」と述べたが、これは誤り。過塩素酸メチルのエステル酸素も
Cl と結合しているため、結合数だけでは区別できない。実際に区別できるのは終端かどうかで、
エステル酸素は C とも結合していて degree 2 になる。

修正後の挙動:

| 分子 | 戻り値 | 分子への変更 |
| --- | --- | --- |
| `[F][P-](F)(F)(F)(F)F` | `{'PF6_P': [1], 'PF6_F': [0,2,3,4,5,6]}` | 意図どおり書き換え |
| `[O-][Cl+3]([O-])([O-])[O-]` | `{'ClO4_Cl': [1], 'ClO4_O': [0,2,3,4]}` | 意図どおり書き換え |
| `[O-]P(=O)(O)O` / `[O-]P(=O)(F)F` / `[O-]P(=O)([O-])[O-]` | `None` | **無変更** |
| `CO[Cl+3]([O-])([O-])[O-]` | `None`（誤検出解消） | **無変更** |

旧コードとの等価性チェック 31 項目は引き続き一致しており、本来の PF6 / ClO4 / FSA の
検出結果は変わっていない。回帰テストを 9 件追加した（該当しない 5 分子について形式電荷・
結合次数・SMILES がすべて不変であること、index の並び、`_central_ion_indices` の終端条件）。
