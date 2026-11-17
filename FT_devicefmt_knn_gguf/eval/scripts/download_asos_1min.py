#!/usr/bin/env python3
"""ASOS 1 分値（NOAA の空港気象観測。Iowa State Univ. の IEM が再配布）を地点 × 年ごとに CSV で取得する。

依存: Python 3.8 以上の標準ライブラリだけ（pandas 不要）。
使い方:
  python3 download_asos_1min.py                                  # 既定: 5 地点 × 2019〜2025 → eval/data/raw/asos_1min/
  python3 download_asos_1min.py --out DIR --stations DSM IAH --years 2024 2025
出力: <out>/iem_<地点>_<年>.csv
  列: station, station_name, valid(UTC), tmpf[°F], dwpf[°F], pres1[inHg], precip[inch/分], ptype[降水種別]
  欠測は "M"。1 年分で 36〜51 万行・25〜35 MB。
  既にあるファイルは飛ばす（再実行しても二重取得しない）。失敗した地点・年は最後にまとめて表示する。
注意: IEM への負荷を考えて 1 本ずつ直列で取得する（1 本 25〜60 秒）。
      プロキシ環境では環境変数 https_proxy を設定しておく（urllib が自動で使う）。
"""
import argparse
import os
import time
import urllib.request

STATIONS = ["DSM", "IAH", "SEA", "ATL", "BOS"]   # デモイン・ヒューストン・シアトル・アトランタ・ボストン
YEARS = list(range(2019, 2026))
VARS = "tmpf,dwpf,pres1,precip,ptype"             # 気温・露点・現地気圧・1 分降水・降水種別
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
       "?station={st}&vars={vars}&sts={y}-01-01T00:00Z&ets={y1}-01-01T00:00Z"
       "&sample=1min&what=download&delim=comma&gis=no&tz=UTC")


def fetch(st, year, out_dir):
    """1 地点 1 年を取得して <out_dir>/iem_<st>_<year>.csv に保存する。成功なら True。"""
    path = os.path.join(out_dir, f"iem_{st}_{year}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        print(f"skip {st} {year} (already exists)")
        return True
    url = URL.format(st=st, vars=VARS, y=year, y1=year + 1)
    for attempt in range(1, 4):
        try:
            t = time.time()
            with urllib.request.urlopen(url, timeout=600) as r, open(path + ".part", "wb") as f:
                f.write(r.read())
            with open(path + ".part", encoding="utf-8") as f:
                first = f.readline()
            if first.startswith("station,"):                 # 正常な CSV のヘッダ
                os.replace(path + ".part", path)
                print(f"ok   {st} {year} {os.path.getsize(path) / 1e6:.1f} MB {time.time() - t:.0f}s")
                return True
            print(f"retry {st} {year}: unexpected response: {first.strip()[:80]}")
        except Exception as e:
            print(f"retry {st} {year}: {e}")
        time.sleep(10 * attempt)
    if os.path.exists(path + ".part"):
        os.remove(path + ".part")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval/data/raw/asos_1min", help="出力ディレクトリ")
    ap.add_argument("--stations", nargs="+", default=STATIONS, help="地点 ID（IEM の ASOS 1 分値の地点一覧にあるもの）")
    ap.add_argument("--years", nargs="+", type=int, default=YEARS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    failed = []
    for st in a.stations:
        for y in a.years:
            if not fetch(st, y, a.out):
                failed.append((st, y))
    print(f"done. failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
