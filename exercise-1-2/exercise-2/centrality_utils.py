from pathlib import Path
import datetime


def print_centrality_results(node_to_centrality, output_dir_path, centrality_name):
    output_file_path = Path(output_dir_path).absolute()
    output_file_path = output_file_path.joinpath(
        f"results_{centrality_name}_{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt")

    node_to_centrality.insert(0, ("Node", centrality_name))
    with open(output_file_path, "w") as output_file:
        for entry in node_to_centrality:
            output_file.write(str(entry[0]) + ":" + str(entry[1]) + "\n")
    return


def load_centrality_results(results_file_path):
    results_file_path = Path(results_file_path).absolute()
    results = []

    with open(results_file_path, "r") as results_file:
        content = results_file.readlines()
        content.pop(0)
        for line in content:
            line = line.strip().split(":")
            results.append((line[0], line[1]))

    return results
