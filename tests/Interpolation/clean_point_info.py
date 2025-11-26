import argparse
import re
import numpy as np


def parse_point_info_line(line):
    """
    Parse one line of *_point_info.txt.

    Expected general shape:
        <index> <type>[x y]...

    We only care about <type>, x, y.
    """
    line = line.strip()
    if not line:
        return None

    # Split off the leading index and then work on the rest of the line.
    # Example line:
    #   "0 C[ 0.41315336 -1.21060971][]"
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None

    rest = parts[1]  # e.g. "C[ 0.41 -1.21][]", "center[-0.37 0.00][1,3,...]", "bond[-0.28 -1.21](0,1)"

    # Match leading type and first [ ... ] block with coordinates
    m = re.match(r"^([A-Za-z0-9_]+)\[([^\]]+)\]", rest)
    if m is None:
        return None

    ptype = m.group(1)
    coord_str = m.group(2)

    # Extract numbers inside the coordinate brackets
    mcoord = re.search(r"^([^\]]+)$", coord_str)
    if mcoord is None:
        return None
    nums = mcoord.group(1).replace(',', ' ').split()
    if len(nums) < 2:
        return None

    try:
        x = float(nums[0])
        y = float(nums[1])
    except ValueError:
        return None

    return ptype, x, y


def clean_point_info(infile, outfile, include_index=False):
    rows = []
    with open(infile, 'r') as fin:
        for line in fin:
            parsed = parse_point_info_line(line)
            if parsed is None:
                continue
            ptype, x, y = parsed
            rows.append((ptype, x, y))
    if not rows:
        raise RuntimeError(f"No valid points parsed from {infile}")

    with open(outfile, 'w') as fout:
        if include_index:
            fout.write("index type x y\n")
            for i, (ptype, x, y) in enumerate(rows):
                fout.write(f"{i} {ptype} {x:.16g} {y:.16g}\n")
        else:
            fout.write("type x y\n")
            for ptype, x, y in rows:
                fout.write(f"{ptype} {x:.16g} {y:.16g}\n")


'''
Use like this:

python clean_point_info.py     data_Mithun/points/OHO-h_1_point_info.txt     data_Mithun/points/OHO-h_1_points_indexed.txt     --include-index
python clean_point_info.py     data_Mithun/points/OHO-h_1_point_info.txt     data_Mithun/points/OHO-h_1_points_clean.txt

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean *_point_info.txt to {type x y} table")
    parser.add_argument("infile", help="Input *_point_info.txt file")
    parser.add_argument("outfile", help="Output text file with columns {type x y} or {index type x y}")
    parser.add_argument("--include-index", action="store_true", help="Prepend point index column")
    args = parser.parse_args()

    clean_point_info(args.infile, args.outfile, include_index=args.include_index)
