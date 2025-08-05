import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import json
from pathlib import Path


def draw_detected_cells(image_color, results):
    for center_x, center_y, radius, _ in results:
        cv2.circle(image_color, (center_x, center_y), radius, (0, 255, 0), 2)
    return image_color


def plot_image(image, num_cells):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.title(f"Matches after NMS: {num_cells}")
    plt.axis("off")
    plt.show()


def show_cells_on_image(color_image, found_cells):
    cells_on_image = draw_detected_cells(color_image, found_cells)
    plot_image(cells_on_image, len(found_cells))


def visualize_hex_lattice_graph(graph, iteration, position, candidates=None, validated=None, figsize=(12, 12)):
    fig, ax = plt.subplots(figsize=figsize)

    node_ids = list(graph.nodes.keys())  # [:max_nodes]
    for node_id in node_ids:
        node = graph.nodes[node_id]
        x, y = node.position
        ax.plot(x, y, "o", color="black", markersize=4)

        for dir_idx, neighbor in node.neighbors.items():
            if neighbor is None:
                vec = (graph.vecs + [-v for v in graph.vecs])[dir_idx]
                end = node.position + vec
                ax.plot([x, end[0]], [y, end[1]], linestyle="dotted", color="gray", alpha=0.3)
            elif neighbor == "OUT_OF_BOUNDS":
                vec = (graph.vecs + [-v for v in graph.vecs])[dir_idx]
                end = node.position + vec
                ax.plot([x, end[0]], [y, end[1]], linestyle="dashed", color="red", alpha=0.5)
            elif neighbor == "CONFLICT":
                vec = (graph.vecs + [-v for v in graph.vecs])[dir_idx]
                end = node.position + vec / 2
                ax.plot([x, end[0]], [y, end[1]], linestyle="dotted", color="cyan", alpha=0.5)
            else:
                if neighbor in graph.nodes:
                    neighbor_node = graph.nodes[neighbor]
                    nx, ny = neighbor_node.position
                    ax.plot([x, nx], [y, ny], color="blue", linewidth=1, alpha=0.7)

    # --- Visualize predicted candidate cells ---
    # if candidates is not None:
    #     for pos, _, _ in candidates:
    #         ax.plot(pos[0], pos[1], "rx", markersize=5)  # Red 'x' for missing neighbor
    if candidates is not None:
        for pos, _, _, method in candidates:
            if method == "curve":
                ax.plot(pos[0], pos[1], marker="x", color="green", markersize=5)
            else:
                ax.plot(pos[0], pos[1], marker="x", color="red", markersize=5)

    if validated is not None:
        for final_pos, support in validated:
            ax.plot(final_pos[0], final_pos[1], marker="*", color="orange", markersize=5)

    ax.set_aspect("equal")
    ax.set_title(f"Iteration: {iteration} - position: {position}")
    ax.invert_yaxis()

    legend_items = [
        mpatches.Patch(color="blue", label="Connected neighbors"),
        mpatches.Patch(color="red", label="Out of bounds (dashed)"),
        mpatches.Patch(color="gray", label="Missing (dotted)"),
        mpatches.Patch(color="green", label="Predicted (curve)"),
        mpatches.Patch(color="red", label="Predicted (lattice)"),
        mpatches.Patch(color="orange", label="Validated (clustered)"),
    ]
    ax.legend(handles=legend_items)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_html_performance_report(results, config, output_dir: Path, timestamped=True) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f"cell_eval_report_{timestamp}.html" if timestamped else "cell_eval_report.html"
    report_path = output_dir / report_name

    html_parts = []

    html_parts.append("<html><head><title>Cell Detection Report</title></head><body>")
    html_parts.append(f"<h1>Cell Detection Evaluation Report ({timestamp})</h1>")

    html_parts.append("<h2>Configuration</h2>")
    html_parts.append("<pre>" + json.dumps(config, indent=4) + "</pre>")

    html_parts.append("<h2>Per-Image Results</h2>")
    for item in results:
        image_name = item["image_name"]
        img = cv2.imread(str(item["img_path"]))
        gt_cells = item["gt_cells"]
        pred_cells = item["pred_cells"]

        pred_img = img.copy()
        gt_img = img.copy()

        for cell in pred_cells:
            center = (int(cell["center_x"]), int(cell["center_y"]))
            radius = int(cell.get("radius", 10))
            cv2.circle(pred_img, center, radius, (0, 0, 255), 2)
        for cell in gt_cells:
            center = (int(cell["center_x"]), int(cell["center_y"]))
            radius = int(cell.get("radius", 10))
            cv2.circle(gt_img, center, radius, (0, 255, 0), 2)

        pred_path = output_dir / f"{timestamp}_{image_name}_pred.png"
        gt_path = output_dir / f"{timestamp}_{image_name}_gt.png"
        cv2.imwrite(str(pred_path), pred_img)
        cv2.imwrite(str(gt_path), gt_img)

        html_parts.append(f"<h3>{image_name}</h3>")
        html_parts.append("<table><tr><td><b>Predictions</b></td><td><b>Ground Truth</b></td></tr>")
        html_parts.append(
            f"<tr><td><img src='{pred_path.name}' width='400'></td><td><img src='{gt_path.name}' width='400'></td></tr></table>"
        )
        html_parts.append("<ul>")
        html_parts.append(f"<li>True Positives: {item['true_positives']}</li>")
        html_parts.append(f"<li>False Positives: {item['false_positives']}</li>")
        html_parts.append(f"<li>False Negatives: {item['false_negatives']}</li>")
        html_parts.append(f"<li>Precision: {item['precision']:.2f}</li>")
        html_parts.append(f"<li>Recall: {item['recall']:.2f}</li>")
        html_parts.append(f"<li>F1 Score: {item['f1_score']:.2f}</li>")
        html_parts.append(f"<li>Mean Localization Error: {item['mean_localization_error']:.2f}</li>")
        html_parts.append("</ul><hr>")

    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    html_parts.append("<h2>Overall Summary</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Total True Positives: {total_tp}</li>")
    html_parts.append(f"<li>Total False Positives: {total_fp}</li>")
    html_parts.append(f"<li>Total False Negatives: {total_fn}</li>")
    html_parts.append(f"<li>Overall Precision: {precision:.2f}</li>")
    html_parts.append(f"<li>Overall Recall: {recall:.2f}</li>")
    html_parts.append(f"<li>Overall F1 Score: {f1:.2f}</li>")
    html_parts.append("</ul>")

    html_parts.append("</body></html>")

    with open(report_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"HTML report saved to: {report_path}")

    return report_name


def save_cell_find_config_to_json(out_path: Path, method: str, **kwargs):
    data = {"method": method}
    data.update(kwargs)

    out_path = out_path / "cell_finder_config.json"
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
