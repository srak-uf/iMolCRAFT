# calculator リファクタリング記録

対象: `imolcraft/calculator/`（psi4geoopt.py / charge.py / distance.py / dihedral.py、計 1603 行）
`energy.py` は未使用のため今回は触っていない（下記）。

## 方針
crafter・analyzer・io と同じ。ファイル構成は維持、挙動は完全維持（等価変換のみ）、
潜在バグは修正せず記録する。

## 検証方法
外部プログラム（g16 / psi4 / antechamber）はこのマシンに無いので、旧コード（git HEAD 版）を
別モジュールとしてロードし、外部実行を伴わない全経路を突き合わせた。

- **純関数・初期化・入力生成: 28 項目一致**
  - `get_rotatable_dihedral` を 5 分子で（二面角の原子 index と元素）
  - `rotate_dihedral` を 5 目標角で（全原子座標が完全一致）
  - `change_distance` を 3 原子対 × 4 目標距離で（全原子座標が完全一致）
  - `load_g16scan` を実ログ `tests/test/test_dihed_0.log` で。角度・エネルギー・
    36 個の構造に加え、**この関数が書き換えるログファイル自体もバイト一致**
  - `DistanceCalculator` / `DihedralCalculator` の初期化（rmin / dr / scan_ranges /
    scan_list / qmparams）
  - `do_qmscan(do_calc=False)` が生成する Gaussian 入力 `.com` が**バイト一致**
- **FF スキャン本体: OpenMM 経路込みで一致**
  実 ffxml（`tests/xmlfiles/ethane1.xml`）とエタンで、`scan_ff_dihedral`（7 点、
  geoopt_atoms 経路）と `scan_ff_distance`（3 点）を実行。角度・距離・エネルギー
  （kJ/mol）・緩和後の全原子座標が一致した。これが最も重く書き換えた 2 関数の直接検証になる。
- `pytest tests -m "not qm"` は 140 passed / 10 failed / 1 error。失敗と error は
  packmol / antechamber / parmchk2 / psi4 の未インストールによる環境要因で、
  リファクタ前と同じ顔ぶれ。

## 主な変更

### 共通（distance.py / dihedral.py）
両ファイルの FF スキャンは同じ形の OpenMM 定型処理を持っていたので、各ファイル内で
ヘルパーに切り出した。

| ヘルパー | 置き換えた内容 |
| --- | --- |
| `_make_integrator()` | `LangevinMiddleIntegrator(300*kelvin, ...)` の 3〜4 重複 |
| `_write_pdb()` | `PDBFile.writeFile(topology, pos, open(path, "w"))`。**ファイルを閉じていなかった**箇所が distance.py に 3、dihedral.py に 3 |
| `_add_bonds()` | トポロジへの結合追加の 2〜3 重複 |
| `_positions_in_angstrom()` | state の座標を angstrom の素の list にする 8 行ループ |
| `_energies_by_force_group()` | 力ごとのエネルギー評価。下記のとおり高速化も兼ねる |
| `_distance_restraint()` / `_dihedral_restraint()` | 仮想サイトを飛ばした index 対応表の構築と拘束力の生成 |

#### `_energies_by_force_group` による高速化

元のコードは力の数だけ `Simulation` を作り直していた。

```python
for j, f in enumerate(system.getForces()):
    f.setForceGroup(j)
    integrator = LangevinMiddleIntegrator(...)
    simulation = Simulation(pdb_omm.topology, system, integrator)   # 毎回作り直し
    simulation.context.setPositions(pdb_omm.positions)
```

ループ内で作った `Simulation` は最後の 1 つしか使われず、途中のものは捨てられる。
`Simulation` の生成は OpenMM の `Context` 生成を伴うので高くつく。力群の設定を先に済ませてから
`Simulation` を 1 つだけ作る形にした。結果は同一で、エタンの 25 点二面角スキャンで
**5.03s → 2.88s（1.7 倍）**。

### psi4geoopt.py
- `generate_input` の 2 つのテンプレート（多原子は `optimize()`、単原子は `energy()`）が
  最終行以外ほぼ同一だったので、`_PSI4_INPUT_TEMPLATE` 1 つに統合し、駆動関数の行だけ
  差し替えるようにした。
- `molecule` ブロックの組み立てを `_molecule_block()` に分離。
- `os.path.join(f"{self.label}.psi4in")` は引数 1 つの `os.path.join` で無意味だったので除去。
- メモリ量とスレッド数の決め方（利用可能メモリを偶数 GB に切り下げ／コア数以下の最大の 2 冪）に
  コメントを付けた。

**唯一のバイト非一致**: 単原子の場合に生成される `.psi4in` の `set { }` ブロックの
インデントが変わる（元は 0 桁、統合後は 3 桁）。psi4 はこのブロックの空白を無視するため
計算結果は同一。多原子の場合はバイト一致。

### charge.py
- `_charges_from_mol2()` 抽出: mol2 の電荷列を読む処理が 3 箇所に重複していた。
- `_resolve_directory()` 抽出: `if directory is None: os.getcwd()` の 2 重複。
- `_run_g16()` 抽出: `_get_resp` から Gaussian 実行部分を分離。
- `Psi4ChargeCalculator._molecule_from_atoms()` 抽出: `__init__` に埋まっていた
  「RDKit 変換 → SDF 書き出し → OpenFF 分子読み込み」を分離。
- マジックリテラル `["esout", "punch", "qout"]` と再試行回数 10 を定数化。
- `resp_params` を書き換える箇所に、後述の副作用を明記する NOTE コメントを追加。

### distance.py
- `do_qmscan` から `_write_g16_input()` を抽出（拘束指定・ase のバグ回避のための
  `.com` 修正・前の点からの構造引き継ぎ）。
- `do_ffscan` の if/else が同じ 3 要素タプルへの代入を 2 度書いていたので、引数だけ選ぶ形に統一。
  同時に、この分岐にあった未使用の `label` / `d1` / `d2` を除去（`label` は
  `f"{self.label}_dihed_{di}"` という distance なのに dihed というコピペ痕だった）。
- `__init__` の rmin / dr 算出をループから内包表記へ。
- `scan_ff_distance` の入れ子になった `if bonds is None` の内側（常に真）を除去。
  未使用の `molecule_list` / `G_list` も除去。
- `ForceField(ffxml)` と `check_vsite(ffxml)` をループ外へ巻き上げ。
- `change_distance` の連結成分探索を `next()` に、`id` という組み込み名のシャドーイングを解消。
  コメントアウトされた旧実装 6 行を削除。

### dihedral.py
- `do_qmscan` から `_write_g16_input()` を抽出。
- `do_ffscan` の角度決定ロジックを `_resolve_angles()` に分離。未使用の
  `label` / `d0`〜`d3` を除去。
- `isinstance(dihed_idx, int or str)` を `isinstance(dihed_idx, int)` に。
  `int or str` は Python では `int` に評価されるので等価変換（下記の潜在バグ参照）。
- `load_g16scan` から `_fix_g16_version_line()` を抽出（cclib のために Gaussian の
  バージョン行を書き換える前処理）。
- `get_rotatable_dihedral` から `_pick_outer_atom()` を抽出。二重ループと
  `break` の入れ子が 20 行あった。
- `scan_ff_dihedral` の 2 分岐が「PDB 書き出し → 結合追加 → 書き戻し」を丸ごと重複して
  いたので、構造を作る部分だけ分岐して以降を共通化。
- `_wrap_deg()` 抽出: `((angle + 180) % 360) - 180` の 3 重複。
- `rotate_dihedral` の未使用変数 `r1` / `r4` を除去、`id` のシャドーイングを解消、
  Rodrigues の回転公式であることをコメントに明記。
- 末尾にあったコメントアウト済みの描画コード約 60 行を削除（`mol_info` の古いキー構造に
  依存しており復活不能。必要なら git 履歴から取れる）。

## 未修正の潜在バグ・気になる点

1. distance.py と dihedral.py の FF スキャンは、ヘルパー 6 個をファイルをまたいで
   重複させている。共通モジュールに置くのが自然だが、ファイル構成維持の方針のため
   各ファイルに複製した。将来 `calculator/_openmm.py` のような置き場を作るとよい。

## 意図的に残しているもの

- `energy.py` は `calculator/__init__.py` から import されておらず参照も 0 件だが、
  デバッグ用として残す方針（2026-08-21 にユーザー確認）。

## テストの新設

`tests/test_calculator/test_pure_functions.py` を新設し、外部プログラムを必要としない
32 件を追加した。既存の `test_calculator` は 4 ファイル中 3 ファイルが g16 / psi4 を要求し、
このマシンでは常に失敗するため、実質的な回帰検出ができていなかった。

- `_wrap_deg`: 境界値 7 通り（180 が -180 に折り返ることを含む）
- `get_rotatable_dihedral`: 5 分子。原子 index が 4 つとも異なること、元素との対応
- `rotate_dihedral`: 6 目標角で実際にその角度になること、結合長が保たれること
- `change_distance`: 3 原子対 × 4 目標距離、および切った結合の反対側が動かないこと
- `load_g16scan`: 実ログで昇順ソートと最小値 0 へのシフト
- `Psi4GeoOptimizer.generate_input`: 多原子で `optimize(`、単原子で `energy(`

## 修正済みバグ

### _get_am1bcc が PDB を書きながら Gaussian 出力形式だと伝えていた（2026-08-21 修正）

```python
write(temp_pdb_name, self.atoms, format="pdb")
cmd_antech = f"antechamber -i {temp_pdb_name} -fi gout ..."
```

`-fi gout` は Gaussian の出力ファイルを表す。書き出しているのは PDB なので `-fi pdb` が正しい。

修正の過程で、その 1 行上の `write(..., format="pdb")` も動かないことが判明した。
ASE 3.23 の PDB 書き出しの形式名は `proteindatabank` で、`pdb` は拡張子のエイリアスに
過ぎず形式名としては拒否される。

```
旧: write(..., format="pdb") -> UnknownFileTypeError: pdb
新: write(..., format="proteindatabank") -> ok
```

つまり `_get_am1bcc` は 1 行目で例外になっており、`-fi gout` の行には到達していなかった。
両方直したので、この関数は初めて動く状態になった（antechamber 本体はこのマシンに無いため、
コマンド文字列までの検証）。

回帰テストは `subprocess.getoutput` を差し替えて antechamber の代わりに mol2 を書かせ、
組み立てられたコマンドに ` -fi pdb ` が含まれ `gout` が含まれないこと、電荷が読めることを確認する。

### scan_ff_dihedral が仮想サイトを無視していた（2026-08-21 修正）

system・拘束力・書き出しは仮想サイトを含む `topology` から作る一方、`Simulation` と
その初期座標だけが仮想サイトを持たない `pdb_omm.topology` / `pdb_omm.positions` だった。
粒子数が食い違うため、仮想サイトのある力場では必ず失敗する。実測:

```
=== vsite_average2.xml (仮想サイト 2 個)
  旧: OpenMMException: Called setPositions() on a Context with the wrong number of positions
  新: 成功 angles=[-60. 0. 60.] E=[4.8e-03 2.29626e+01 0.0]  緩和後の原子数=10
=== vsite_average3.xml (仮想サイト 1 個)
  旧: OpenMMException: Called setPositions() on a Context with the wrong number of positions
  新: 成功
```

`Modeller` が返す座標（元のコードでは `# pos = modeller.getPositions()` とコメントアウト
されていた）を使い、system・拘束力・`Simulation`・最小化後の書き出しをすべて `topology` /
`pos` に統一した。`distance.scan_ff_distance` は元からこの形になっており、それに揃えた形。

あわせて、次の反復に渡す `pos_prev` から仮想サイトを除く `_real_atom_positions_in_angstrom`
を追加した。`ase.Atoms` は実原子しか持たないため、除かないと座標の代入で形状が食い違う。
同じ対応を `distance.scan_ff_distance` にも適用した（そちらは topology の扱いは元から
正しく、座標の絞り込みだけが欠けていた）。実測:

```
=== vsite_average2.xml (仮想サイト 2 個)
  scan_ff_dihedral 旧: OpenMMException: Called setPositions() ... wrong number of positions
  scan_ff_dihedral 新: 成功 原子数=10
  scan_ff_distance 旧: ValueError: could not broadcast input array from shape (12,3) into shape (10,3)
  scan_ff_distance 新: 成功 原子数=10
=== vsite_average3.xml (仮想サイト 1 個)
  scan_ff_distance 旧: ValueError: ... from shape (11,3) into shape (10,3)
  scan_ff_distance 新: 成功 原子数=10
```

仮想サイトが無い場合は `topology` と `pdb_omm.topology` が同一なので挙動は変わらない。
実際、仮想サイト無しの ffxml による等価性チェック（`scan_ff_dihedral` 7 点 /
`scan_ff_distance` 3 点）は引き続き一致している。

### assert によるバリデーション（2026-08-21 修正）

`assert` は `python -O` で除去されるため入力検証に使えない。`dihedral.py` の 3 箇所を
`raise ValueError` に置き換えた。

| 箇所 | 検出する条件 |
| --- | --- |
| `scan_ff_dihedral` | `atoms_list` と `geoopt_atoms` がどちらも None |
| `scan_ff_dihedral` | `angles` と `atoms_list` の長さ不一致（実際の長さをメッセージに含めた） |
| `rotate_dihedral` | 分子の切り出しに失敗（`raise ... from exc` で元の例外を保持し、error.pdb に書き出した旨をメッセージに追加） |

これで calculator 内の `assert` は 0 件になった。

### Psi4ChargeCalculator が directory=None を受け付けなかった（2026-08-21 修正）

`directory` の正規化（None ならカレントディレクトリ）が、それを使う場所より**後**にあった。

```python
self.molecule = self._molecule_from_atoms(atoms, netcharge, label, directory)  # ここで使う
...
self.directory = _resolve_directory(directory)                                 # 正規化はここ
```

`_molecule_from_atoms` の中で `os.path.exists(directory)` を呼ぶため、`directory=None` だと
`TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType` になる。
docstring と基底クラスの両方が None を許容すると書いているのに、実際には使えなかった。

正規化を `__init__` の 1 行目に移し、以降は `self.directory` を使うようにした。

```
旧: TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
新: 成功 directory='/private/var/.../tmp9dr7xot9' sdf=True
```

これで None のときの挙動が基底クラスと自動的に揃い、SDF の出力先と `self.directory` が
必ず一致する（従来は別々に `directory` を参照していたので、片方だけ変えると静かにずれた）。
あわせて `if not os.path.exists(...): os.makedirs(...)` を
`os.makedirs(..., exist_ok=True)` にした（競合状態の余地が無く、意図が 1 行で出る）。

現状 `TypeError` で落ちる経路なので、動いている呼び出しは影響を受けない。
`Crafter.get_partial_charges` は常に `info["directory"]` を渡すため実運用では到達しない。

回帰テストを 2 件追加（`directory=None` でカレントディレクトリに SDF が出ること、
存在しない入れ子ディレクトリが作られること）。

## 今後の課題

ディレクトリを作るかどうかがクラス間でばらついている。`Psi4ChargeCalculator` と
`DistanceCalculator` は作るが、`ChargeCalculator` と `DihedralCalculator` は保存するだけで
作らない。後者に存在しない出力先を渡すと書き込み時に `FileNotFoundError` になる。
`_resolve_directory` に作成も担わせて統一するのが素直だが、4 クラスすべての挙動が変わるため
別件として残した。

### ChargeCalculator がモジュールレベルの resp_params を破壊していた（2026-08-21 修正）

```python
resp_params = {"method": "hf", "basis": "6-31g(d)", ...}   # モジュールレベル

if self.params is None:
    self.params = resp_params          # 参照を共有
else:
    resp_params.update(self.params)    # モジュール変数をその場で書き換え
    self.params = resp_params          # 参照を共有
```

独立した問題が 2 つ重なっていた。

1. `dict.update()` は新しい辞書を返さず、呼ばれた辞書自身を書き換える。
   「既定値にユーザー指定を上書きしたものを得る」という意図に対して手段が破壊的だった。
2. `self.params = resp_params` は辞書をコピーせず同じオブジェクトを指すため、
   問題 1 が無くても全インスタンスが同一の辞書を共有していた。

修正前の実測:

```
起動直後の resp_params["method"]: hf
A を params={"method":"b3lyp"} で作った後
  m.resp_params["method"]   : b3lyp   <- 既定値が書き換わった
B を params 未指定で作った後
  B.params["method"]        : b3lyp   <- hf のはずが b3lyp
  A.params is B.params      : True
  A.params is m.resp_params : True
B.params["basis"] を書き換えると
  A.params["basis"]         : sto-3g  <- 別インスタンスまで巻き添え
```

`self.params` は `Gaussian(label=..., charge=..., **self.params)` に渡されるので、
漏れていたのは**計算手法そのもの**。既定の HF で RESP 電荷を出すつもりの計算が黙って
B3LYP になる。例外は出ないため、結果を見比べない限り気づけない。

既定値とユーザー指定を新しい辞書にマージする `_merged_resp_params()` を追加し、
両方の分岐をそれ経由にした。`ioplist` のような入れ子の可変値まで切り離すため deep copy を使う。

```python
def _merged_resp_params(params=None):
    merged = copy.deepcopy(resp_params)
    if params is not None:
        merged.update(copy.deepcopy(params))
    return merged
```

修正後の実測:

```
起動直後                : hf 6-31g(d)
A(params=b3lyp) 作成後  : resp_params は hf のまま / A.params: b3lyp 6-31g(d)
B(params 未指定) 作成後 : B.params: hf   A.params is B.params: False
B を書き換えた後        : resp_params も A.params も無傷
```

上書きしなかった既定値（`basis` など）が残るマージ動作は従来どおり。挙動が変わるのは
「複数インスタンスで params が混ざっていた」ケースだけで、そこは元々バグ。
単独インスタンスの結果は変わらない。

なお `Psi4ChargeCalculator` は `self.params = ESPSettings(...)` と毎回新しいオブジェクトを
作るため、この問題は無かった。同じ「既定値とユーザー指定のマージ」を 2 クラスが別々の方法で
書いていて、片方だけが壊れていた形。

回帰テストを 3 件追加（既定値が変わらないこと、インスタンス間で辞書を共有しないこと、
入れ子のリストまで独立していること）。

### parse_g16scan の削除（2026-08-21）

バグ修正ではなく死んだコードの除去。`dihedral.parse_g16scan` はリポジトリ内のどこからも
参照されていなかった（`.py` / `.ipynb` / `.md` を全走査して 0 件）。

```python
for line in lines:
    if "Optimization completed." in line:
        scanned_energy.append(parserd_lines[-1])
        parserd_lines.append(line)          # str を追加
    elif "SCF Done:" in line:
        parserd_lines.append(float(...))    # float を追加
```

`parserd_lines` に str と float を混在させており、最初に現れる行が
"Optimization completed." だと空リストの `[-1]` で `IndexError` になる。
同じ役割は cclib を使う `load_g16scan` が担っており、そちらは実ログでの回帰テストもある。

`from .dihedral import *` により `imolcraft.calculator.parse_g16scan` として公開されていた
ため、**破壊的変更**にあたる。リポジトリ外のコードが参照している場合は
`AttributeError` になる。削除後の公開名は
`change_distance` / `get_rotatable_dihedral` / `load_g16scan` / `rotate_dihedral` /
`scan_ff_dihedral` / `scan_ff_distance` の 6 つ。

### do_ffscan が文字列の index と未知の angles を扱えなかった（2026-08-21 修正）

```python
if isinstance(dihed_idx, int or str):
    dihed_idx = [int(dihed_idx)]
```

`int or str` は「型の集合」ではなく普通のブール演算式として先に評価される。`or` は左辺が
真ならそれを返し、クラスオブジェクト `int` は真なので、式全体が `int` になる。

```
int or str        -> <class 'int'>
(int, str)        -> (<class 'int'>, <class 'str'>)
  isinstance('0', int or str) = False   isinstance('0', (int, str)) = True
```

つまりこの行は最初から `isinstance(dihed_idx, int)` と等価で、`str` の分岐は存在していなかった。
リファクタ時に `isinstance(dihed_idx, int)` と書き換えたのは実態を写しただけで挙動は不変。

正規化を素通りした文字列はその下の `for di in dihed_idx:` に渡る。文字列はイテラブルなので
エラーにならず 1 文字ずつ回り、`self.qm_scan[di]` で落ちる。`"12"` のような 2 桁を渡すと
「12 番目」ではなく「1 番目と 2 番目」を意図した動作になりかけたうえで型エラーになる。

`distance.do_ffscan` は `(int, str)` と正しく書かれていたので、それに揃えた。

```
  旧 dihed_idx=0    -> scan 呼び出し [[1, 0, 4, 5]]
  旧 dihed_idx='0'  -> TypeError: list indices must be integers or slices, not str
  新 dihed_idx=0    -> scan 呼び出し [[1, 0, 4, 5]]
  新 dihed_idx='0'  -> scan 呼び出し [[1, 0, 4, 5]]
```

あわせて `_resolve_angles` の同種の穴も塞いだ。`isinstance` の使い方自体は正しかったが、
`"QM"` 以外の文字列がそのまま角度列として返っていた。`scan_ff_dihedral` の `len(angles)` は
文字数を数えるので、静かにおかしな動作になる。

```
  新 angles='QM'  -> []（QM スキャン結果）
  新 angles='FF'  -> ValueError: angles must be "QM" or a sequence of angles in degrees: 'FF'
```

どちらも受け付ける入力が広がる／エラーが早くなる方向の変更で、従来動いていた呼び出しの
結果は変わらない。回帰テストを 9 件追加した（`dihed_idx` の int / str / list、
`_resolve_angles` の 4 系統、未知文字列 3 種）。

### _pick_outer_atom が None を返しうる形だった（2026-08-21 修正）

二面角の外側の参照原子が見つからないとき、リファクタ後の `_pick_outer_atom` は `None` を
返し、`get_rotatable_dihedral` の戻り値に `None` が混ざる形になっていた。

元のコードはもう少し悪く、`d0` / `d3` を条件分岐の中でしか代入していなかったため、
見つからない場合は変数が未定義になる。しかも関数スコープなので、2 本目以降の回転可能結合を
処理しているときは**前の結合の値がそのまま残って使われる**。エラーにならず別の原子で
二面角が定義されてしまう。

まず到達可能性を調べた。回転可能結合の SMARTS

```
[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]
```

は両端に `!D1`（次数 1 でない）を課している。一致した結合そのものが 1 本を占めるので、
両端とも必ず別の隣接原子を持つ。13 分子 × 明示水素あり/なしで全一致を走査した結果、
外側原子が無いケースは 0 件だった。つまりこの経路は SMARTS 経由では到達しない。

そこで `None` を返す代わりに、不変条件が破れたことを示す `ValueError` を送出する形にした。

```python
raise ValueError(
    f"Atom {atom.GetIdx()} of the rotatable bond {d1}-{d2} has no other "
    "neighbour, so no dihedral can be defined around that bond"
)
```

「SMARTS 一致に対しては起きえず、独自の結合リストを渡す呼び出し元だけを守るガードである」
ことを docstring に明記した。黙って `None` を流すと、`DihedralCalculator` の
`d0 + 1` で `TypeError` になるまで原因が見えない。

シグネチャも `_pick_outer_atom(bonds, d1, d2)` から `(atom, d1, d2)` に変え、
エラーメッセージに原子番号を出せるようにした。

回帰テストを 25 件追加（末端原子で `ValueError` になること、13 分子 × 明示水素あり/なしで
戻り値に `None` が混ざらず 4 原子がすべて異なること）。
