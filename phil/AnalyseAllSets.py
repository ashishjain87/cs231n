import argparse
import os
from pathlib import Path
from typing import List

# PATH_TO_TEST_DIR = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5/'
def create_yaml(data_dir: Path, sets: List, batch_size: int, weights_path: Path, model_name: str, baseline_exp_dir: Path, baseline_name: str, analysis_dir: Path, num_examples_to_visualise: int, show_plot: bool):
    for set in sets:
        yaml_path = Path(analysis_dir, "test.yaml")
        with open(yaml_path, 'w') as yaml_file:
            contents = f"test: {data_dir}/images/{set}\n\nnc: 3\nnames: ['vehicle', 'person', 'misc']"
            yaml_file.write(contents)

        # os.system(f"python ../yolov5/test.py --img 640 --name {model_name}-{set} --batch-size {batch_size} --data {yaml_path} --weights {weights_path} --task test --save-hybrid --save-conf --conf-thres 0.25 --iou-thres 0.6")
        baseline_dir_str = f"--baseline-exp-dir {Path(baseline_exp_dir)} " if baseline_exp_dir is not None else ""
        show_plot_str = f"--show-plot " if show_plot else ""

        os.system(f"python ./phil/analyse.py --data {data_dir} " +
                        f"--yolo ../yolov5 " +
                        f"--model-exp-dir ./runs/test/{model_name}-{set} " +
                        f"--model-name {model_name} " + 
                        baseline_dir_str +
                        f"--baseline-name {baseline_name} " + 
                        f"--analysis-dir {analysis_dir} " +
                        f"--split-name {set} " +
                        f"--num-examples-to-visualise {num_examples_to_visualise} " +
                        show_plot_str
                    )

if __name__ == "__main__":
    """
    Example Usage:
    python ./phil/AnalyseAllSets.py \
        --data-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/fixed_final/ \
        --sets 'val-side-affixer-different-class' \
        --batch-size 1 \
        --model-name baseline-amodal-combined-train \
        --weights ../baseline-amodal-combined-train/weights/best.pt \
        --baseline-name baseline \
        --baseline-exp-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5/runs/test/exp41/ \
        --analysis-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/analysis-baseline-amodal-combined-train \
        --num-examples-to-visualise 0 \
        --show-plot

    Recommendation:
        - Call the analysis directory something to do with the model you are running, e.g.
            /Users/philipmateopfeffer/Downloads/cs231n_class_project/analysis-baseline-amodal-combined-train
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None, help='top level directory of the sets you want to run inference on')
    parser.add_argument('--sets', type=str, default=None, help='string of comma separated sets you want to run inference on')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--weights', type=str, default=None, help='weights path')
    parser.add_argument('--model-name', type=str, default=None, help='model name')
    parser.add_argument('--show-plot', action='store_true', help='show the plot in matplotlib')
    parser.add_argument('--analysis-dir', type=str, default=None, help='dir to save plot')
    parser.add_argument('--baseline-name', type=str, default=None, help='baseline model name')
    parser.add_argument('--baseline-exp-dir', type=str, default=None, help='dir to baseline experiment runs/exp<NUM>')
    parser.add_argument('--num-examples-to-visualise', type=int, default=5, help='num examples to visualise (non-deterministic)')
    args = parser.parse_args()
    args.sets = [set.strip() for set in args.sets.split(',')]
    
    print("[WARNING] Make sure you've run this code on the baseline first!")
    print("[RECOMMENDED] Call the analysis directory something to do with the model you are running, e.g. /home/project/analysis-baseline-amodal-combined-train")
    if args.analysis_dir is None:
        raise ValueError("Must specify analysis dir path")
    if args.model_name is None:
        raise ValueError("Must specify model name")
    if (args.baseline_name is None) ^ (args.baseline_exp_dir is None):
        raise ValueError("Must specify both --baseline-name and --baseline-exp-dir")
    
    if args.analysis_dir is not None and not os.path.exists(Path(args.analysis_dir, "histograms")):
        os.makedirs(Path(args.analysis_dir, "histograms"))

    create_yaml(Path(args.data_dir), args.sets, args.batch_size, Path(args.weights), args.model_name, args.baseline_exp_dir, args.baseline_name, Path(args.analysis_dir), args.num_examples_to_visualise, args.show_plot)

