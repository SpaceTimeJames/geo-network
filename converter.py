import itn
import csv
import os
import operator


def gml_to_node_edge_list(infile, outfile=None, routing=False, write_to_disk=True):
    """
    Convert a GML file to a list with the following format per row
    start_node, end_node, edge_id, edge_length
    :param infile: Input gml file
    :param outfile: If supplied, this is used for saving. Otherwise the output file is the same as input but with the
    extension .csv added
    :param routing: If True, respect routing restrictions
    :param write_to_disk: If True, save the output, otherwise just return the data.
    :return: List of (start_node, end_node, edge_id, edge_length) tuples
    """

    if outfile is None:
        outfile = infile + '.csv'
    if os.path.exists(outfile):
        raise ValueError("Output file %s already exists", outfile)

    # read gml
    data = itn.read_gml(infile)
    net = itn.ITNStreetNet.from_data_structure(data)

    g = net.g_routing if routing else net.g
    out_data = []

    edges_seen = set()
    for start_node in g.edge.iterkeys():
        for end_node, edges in g.edge[start_node].iteritems():
            for edge_id, attr in edges.items():
                if routing:
                    if (start_node, end_node, edge_id) in edges_seen:
                        continue
                else:
                    if (start_node, end_node, edge_id) in edges_seen or (end_node, start_node, edge_id) in edges_seen:
                        continue

                edges_seen.add((start_node, end_node, edge_id))
                t = (start_node, end_node, edge_id, attr['length'])
                out_data.append(t)

    fields = (
        'start_node',
        'end_node',
        'edge_id',
        'edge_length',
    )

    with open(outfile, 'wb') as f:
        c = csv.writer(f)
        c.writerow(fields)
        c.writerows(out_data)

    return out_data
