from converter import gml_to_node_edge_list
import sys


if __name__ == '__main__':
    in_file = sys.argv[1]
    res = gml_to_node_edge_list(in_file, routing=True)
