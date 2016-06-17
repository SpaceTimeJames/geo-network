import sys
import os
sys.path.append(os.path.abspath(os.path.curdir))

from converter import gml_to_node_edge_list


if __name__ == '__main__':
    in_file = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else None
    res = gml_to_node_edge_list(in_file, outfile=outfile, routing=True)
