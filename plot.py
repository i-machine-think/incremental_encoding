import matplotlib.pyplot as plt
from custom_logging import LogCollection
import re

#######################################################
# Example file that illustrates the use of log plotting

def name_parser(filename, subdir):
    splits = filename.split('/')
    return splits[1]+'_'+splits[-2]


############################
# helper funcs

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def func(input_str):
    if 'full_focus' in input_str and 'hard' not in input_str and 'baseline' not in input_str:
        return True
    return False

def f64_256(input_str):
    if 'E64xH128' in input_str and 'run_1' in input_str:
        return True
    return False

def pre_rnn(input_str):
    if 'pre_rnn' in input_str\
            and 'baseline' not in input_str and 'hard' not in input_str:
        return True
    return False

def full_focus(input_str):
    if 'full_focus' in input_str\
            and 'baseline' not in input_str and 'hard' not in input_str:
            # and 'E64xH512' in input_str:
        return True
    return False

def pre_ff_baseline(input_str):
    if 'hard' not in input_str:
        return True
    return False

def ff_and_baseline(input_str):
    if ('focus' in input_str and 'baseline' in input_str) or \
            ('focus' in input_str and 'hard' not in input_str):
        return True
    return False

def pre_and_baseline(input_str):
    if 'pre_rnn' in input_str and 'hard' not in input_str\
            and 'H16' not in input_str and 'H32' not in input_str:
        return True
    return False

def best_pre_and_baseline(input_str):
    if 'pre_rnn' in input_str and ( \
            ('hard' not in input_str and 'E16xH512' in input_str \
            and 'baseline' not in input_str) \
            or ('baseline' in input_str and 'E128xH512' in input_str)):
        return True
    return False

def hard(input_str):
    if 'hard' in input_str and 'pre_rnn' in input_str:
        return True
    return False

def baseline(model):
    if 'baseline' in model and 'pre_rnn' in model:
            # and ('E16xH256' in model or 'E62xH256'  in model or 'E64xH512' in model):
        return True
    return False

def data_name_parser(data_name, model_name):
    if 'Train' in data_name and 'baseline' in model_name:
        label = 'Baseline, training loss'
    elif 'Train' in model_name:
        label = 'Attention Guidance, Train'
    elif 'baseline' in model_name:
        label = 'Baseline, test loss'
    else:
        label = 'Attention Guidance, test loss'
    return label

def heldout_tables(input_str):
    if 'heldout_tables' in input_str:
        return True
    return False

def heldout_inputs(input_str):
    if 'heldout_inputs' in input_str:
        return True
    return False

def heldout_compositions(input_str):
    if 'heldout_compositions' in input_str:
        return True
    return False

def not_longer(input_str):
    if 'longer' not in input_str:
        return True
    return False

def not_train(dataset):
    if 'Train' not in dataset:
        return True
    return False

def color_train(model_name, data_name):
    if 'Train' in data_name and 'baseline' in model_name:
        c = 'k--'
    elif 'Train' in data_name:
        c = 'k'
    elif 'baseline' in model_name:
        c = 'm:'
    else:
        c = 'g'

    return c

def color_groups(model_name, data_name):
    if 'baseline' in model_name:
        c = 'b'
    elif 'hard' in model_name:
        c = 'm'
    else:
        c = 'g'

    if 'pre_rnn' in model_name:
        l = ':'
    elif 'full_focus' in model_name:
        l = '-'
    elif 'post_rnn' in model_name:
        l = '--'

    return c+l

def find_basename(model_name):
    all_parts = model_name.split('_')
    basename = '_'.join(all_parts[2:])
    return basename

def no_basename(model_name):
    return model_name

def find_data_name(dataset):
    dataname = dataset.split('/')[-1].split('.')[0]
    if 'longer' in dataname:
        splits = dataname.split('_')
        elements = [splits[0],splits[2]]
        dataname = '_'.join(elements)
    return dataname

def color_baseline(model_name, data_name):
    if 'baseline' in model_name:
        c = 'm'
    else:
        c = 'g'
    return c

def color_conditions(model_name, data_name):
    if 'baseline' in model_name:
        c = 'm'
    elif 'focus' in model_name:
        c = 'b'

    if 'Train' in data_name:
        c = 'k'
        l = '-'
    elif 'inputs' in data_name:
        l = '-'
    elif 'tables' in data_name:
        l = '--'
    elif 'compositions' in data_name and 'heldout' in data_name:
        l = '-.'
    elif 'new' in data_name:
        l = ':'

    return c+l

def color_size(model_name, data_name):
    if 'H16' in model_name:
        c = 'b'
    elif 'H32' in model_name:
        c = 'g'
    elif 'H64' in model_name:
        c = 'k'
    elif 'H128' in model_name:
        c = 'r'
    elif 'H256' in model_name:
        c = 'm'
    elif 'H512' in model_name:
        c = 'c'
    return c


#log = LogCollection()
#log.add_log_from_folder('chosens_dump', ext='.dump', name_parser=name_parser)

#fig = log.plot_groups('nll_loss', restrict_model=ff_and_baseline, find_basename=find_basename, find_data_name=find_data_name, restrict_data=not_longer, color_group=color_conditions, eor=-135)
#fig.savefig('/home/dieuwke/Documents/papers/AttentionGuidance/figures/lookup_loss_convergence.png')
# plot_pre_and_baseline()

# plot_heldout_tables_all()
# plot_size_correlation()
# plot_val_loss()


def custom_name_parser(filename, subdir):
    splits = filename.split('/')
    return splits[-1].replace(".log", "")


if __name__ == "__main__":
    LOG_PATH = "../logs/"
    lc = LogCollection()

    lc.add_log_from_folder(LOG_PATH, name_parser=custom_name_parser, ext=".log")

    # Plot different losses
    #fig = lc.plot_groups("nll_loss", find_basename=lambda name: name.replace("_", " "), find_data_name=lambda name: name.lower(), eor=-1000)
    #fig = lc.plot_metric("nll_loss", title="NLLoss on SCAN lengths", restrict_model=lambda name: "lengths" in name)
    #fig = lc.plot_metric("nll_loss", title="NLLoss on SCAN left", restrict_model=lambda name: "left" in name)
    #fig = lc.plot_metric("nll_loss", title="NLLoss on SCAN jump", restrict_model=lambda name: "jump" in name)
    #fig = lc.plot_metric("nll_loss", title="NLLoss on SCAN", restrict_model=lambda name: "simple" in name)
    def color_datasets(name, datasets):
        if "simple" in name:
            return "orange"
        if "lengths" in name:
            return "green"
        if "jump" in name:
            return "purple"
        if "left" in name:
            return "cyan"

    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss for Vanilla Model across Datasets", eor=-100,
    #     restrict_model=lambda name: "incremental" in name and "scaled" not in name and "attention" not in name,
    #     color_group=color_datasets, linestyle = "-", alpha = 0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/antcp_loss.png")

    # Does scaling the anticipation loss make a difference for convergence?
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Simple Split", restrict_model=lambda name: "simple" in name and "attention" not in name,
    #     color_group=lambda name, _: "red" if "scaled" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/simple_scaling.png")
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Lengths Split",
    #     restrict_model=lambda name: "lengths" in name and "attention" not in name,
    #     color_group=lambda name, _: "red" if "scaled" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/lengths_scaling.png")
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Jump Split",
    #     restrict_model=lambda name: "jump" in name and "attention" not in name,
    #     color_group=lambda name, _: "red" if "scaled" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/jump_scaling.png")
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Left Split",
    #     restrict_model=lambda name: "left" in name and "attention" not in name,
    #     color_group=lambda name, _: "red" if "scaled" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_scaling.png")

    # Does attention male a difference for convergence
    # NLL Loss
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Simple Split",
    #     restrict_model=lambda name: "simple" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/simple_nll_attn.png")
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Lengths Split",
    #     restrict_model=lambda name: "lengths" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/lengths_nll_attn.png")
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Jump Split",
    #     restrict_model=lambda name: "jump" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/jump_nll_attn.png")
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLoss Left Split",
    #     restrict_model=lambda name: "left" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_nll_attn.png")
    #
    # # Anticipation loss
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Simple Split",
    #     restrict_model=lambda name: "simple" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/simple_antcp_attn.png")
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Lengths Split",
    #     restrict_model=lambda name: "lengths" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/lengths_antcp_attn.png")
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Jump Split",
    #     restrict_model=lambda name: "jump" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/jump_antcp_attn.png")
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Left Split",
    #     restrict_model=lambda name: "left" in name and "scaled" not in name,
    #     color_group=lambda name, _: "red" if "attention" in name else "blue", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_antcp_attn.png")

    # Extreme scaling experiments
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLLoss Left Scaled=0.5",
    #     restrict_model=lambda name: "left" in name and "scaled" in name and "3" not in name,
    #     color_group=lambda name, _: "green", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_nll_05.png")
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Left Scaled=0.5",
    #     restrict_model=lambda name: "left" in name and "scaled" in name and "3" not in name,
    #     color_group=lambda name, _: "purple", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_antcp_05.png")

    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLLoss Left Scaled=0.01",
    #     restrict_model=lambda name: "left" in name and "001" in name,
    #     color_group=lambda name, _: "green", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_nll_001.png")
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Left Scaled=0.01",
    #     restrict_model=lambda name: "left" in name and "001" in name,
    #     color_group=lambda name, _: "purple", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_antcp_001.png")
    #
    # fig = lc.plot_metric(
    #     "nll_loss", title="NLLLoss Left Scaled=10",
    #     restrict_model=lambda name: "left" in name and "10" in name,
    #     color_group=lambda name, _: "green", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_nll_10.png")
    # fig = lc.plot_metric(
    #     "antcp_loss", title="Anticipation Loss Left Scaled=10",
    #     restrict_model=lambda name: "left" in name and "10" in name,
    #     color_group=lambda name, _: "purple", linestyle="-", alpha=0.3
    # )
    # plt.savefig("/Users/dennisulmer/Desktop/left_antcp_10.png")

    # Plot losses and accuracies for scaling experiments on test and train
    def clean_log_data(log_collection):
        for log in log_collection.logs:
            for key, value in log.data.items():
                if "tasks_test_addprim_turn_left.txt" in key:
                    log.data["Test"] = value
                    del log.data[key]

        return log_collection

    def restrict_left_scaling_05(name):
        return "left" in name and "scaled" in name and "amsgrad" not in name and "primleft_" in name

    def distinguish_test_and_train_nll(name, group):
        return "lime" if "Test" in group else "forestgreen"

    def distinguish_test_and_train_antcp(name, group):
        return "violet" if "Test" in group else "purple"

    def distinguish_test_and_train_seqacc(name, group):
        return "lightcoral" if "Test" in group else "maroon"

    lc = clean_log_data(lc)

    fig = lc.plot_metric(
        "nll_loss", title="NLLLoss Left Scaled=0.5",
        restrict_model=restrict_left_scaling_05,
        color_group=distinguish_test_and_train_nll, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_nll_05.png")
    fig = lc.plot_metric(
        "antcp_loss", title="Anticipation Loss Left Scaled=0.5",
        restrict_model=restrict_left_scaling_05,
        color_group=distinguish_test_and_train_antcp, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_antcp_05.png")
    fig = lc.plot_metric(
        "seq_acc", title="Sequence Accuracy Left Scaled=0.5",
        restrict_model=restrict_left_scaling_05,
        color_group=distinguish_test_and_train_seqacc, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_seqacc_05.png")

    fig = lc.plot_metric(
        "nll_loss", title="NLLLoss Left Scaled=100",
        restrict_model=lambda name: "primleft100" in name,
        color_group=distinguish_test_and_train_nll, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_nll_100.png")
    fig = lc.plot_metric(
        "antcp_loss", title="Anticipation Loss Left Scaled=100",
        restrict_model=lambda name: "primleft100" in name,
        color_group=distinguish_test_and_train_antcp, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_antcp_100.png")
    fig = lc.plot_metric(
        "seq_acc", title="Sequence Accuracy Left Scaled=100",
        restrict_model=lambda name: "primleft100" in name,
        color_group=distinguish_test_and_train_seqacc, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_seqacc_100.png")

    fig = lc.plot_metric(
        "nll_loss", title="NLLLoss Left Scaled=0.01",
        restrict_model=lambda name: "primleft001" in name,
        color_group=distinguish_test_and_train_nll, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_nll_001.png")
    fig = lc.plot_metric(
        "antcp_loss", title="Anticipation Loss Left Scaled=0.01",
        restrict_model=lambda name: "primleft001" in name,
        color_group=distinguish_test_and_train_antcp, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_antcp_001.png")
    fig = lc.plot_metric(
        "seq_acc", title="Sequence Accuracy Left Scaled=0.01",
        restrict_model=lambda name: "primleft001" in name,
        color_group=distinguish_test_and_train_seqacc, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/left_seqacc_001.png")

    # Plot models with only nll-loss or anctp-loss
    fig = lc.plot_metric(
        "nll_loss", title="NLLLoss Only",
        restrict_model=lambda name: "nlll_only" in name,
        color_group=distinguish_test_and_train_nll, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/nll_only_loss.png")
    fig = lc.plot_metric(
        "seq_acc", title="Sequence Accuracy NLLLoss Only",
        restrict_model=lambda name: "nlll_only" in name,
        color_group=distinguish_test_and_train_seqacc, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/nll_only_seqacc.png")

    fig = lc.plot_metric(
        "antcp_loss", title="Anticipation Loss Only",
        restrict_model=lambda name: "antcp_only" in name,
        color_group=distinguish_test_and_train_antcp, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/antcp_only_loss.png")
    fig = lc.plot_metric(
        "seq_acc", title="Sequence Accuracy Anticipation Loss Only",
        restrict_model=lambda name: "antcp_only" in name,
        color_group=distinguish_test_and_train_seqacc, show_figure=False, linestyle="-", alpha=0.3
    )
    plt.savefig("/Users/dennisulmer/Desktop/antcp_only_seqacc.png")
