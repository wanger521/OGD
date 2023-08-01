import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config


def draw_all(plott='accregret', descent_way='model', m=0, eta=0.1, setting='iid', mode='default', groot = 'test',
             draw_iter=Config.ByrdSagaConfig['iterations']):
    """
    Draw the curve of experimental results

    :param attack:
                   '': without Byzantine attacks,
                   'sf': sign-flipping attacks
                   'gs': gaussian attacks
                   'hd': sample-duplicating attacks
    """

    iter = [i+1 for i in range(draw_iter)]

    last_str = ['', '-sf', '-gs', '-hd']
    attack_name = ['without attack', 'sign-flipping attack', 'Gaussian attack', 'sample-duplicating attack']
    algorithms = []
    colors = []
    markers = []
    labels = []
    if mode == 'default':
        algorithms = ['Mean', 'CMedian', 'trimmed-mean', 'GeoMed',  'Krum', 'CenterClip', 'Phocas', 'FABA']
        # colors = ['black','gold', 'skyblue', 'brown', 'olive', 'blue', 'darkgray', 'purple']
        markers = ['h', '1', 'v', 'o', 'x', '*', 'd', '+']
        linestyles = ['-', '-.', ':', '--', '-', '-.', ':', '--']
        labels = ['mean', 'coordinate-wise median', 'trimmed mean', 'geometric median', 'Krum', 'centered clipping', 'Phocas', 'FABA']


    acc_list = []
    regret_list = []
    FONT_SIZE = 20
    LEGEND_SIZE = 20
    LABEL_SIZE = 15
    SCALE = 2.5
    for i in range(len(algorithms)):
        acc_alo = []
        regret_alo = []
        for j in range(len(last_str)):
            path_open = '..\\results\\{0}\\{1}\\{2}\\{2}{3}-step{4}-setting-{5}-momentum{6}.pkl'.format(groot,
                            descent_way, algorithms[i], last_str[j], str(int(eta*10)), setting, str(int(m*10)))
            with open(path_open, 'rb') as f:
                acc, _, regret = pickle.load(f)

                if plott =='avgregret':
                    np_r = np.array(regret[:draw_iter])
                    np_i = np.array(iter)
                    regret =list(np.divide(np_r, np_i))
                acc_alo.append(acc[:draw_iter])
                regret_alo.append(regret[:draw_iter])
        acc_list.append(acc_alo)
        regret_list.append(regret_alo)

    # Plot the curve
    rows = []
    plot_list = [acc_list] + [regret_list]
    y_label_list = []
    if plott == 'accregret':
        rows = [0, 1]
        plot_list = [acc_list] + [regret_list]
        y_label_list = ['Classification Accuracy', 'Adversarial Regret']
    elif plott == plott == 'avgregret':
        rows = [0, 1]
        plot_list = [acc_list] + [regret_list]
        y_label_list = ['Classification Accuracy', 'Adversarial Regret/t']

    fig, axs = plt.subplots(len(rows), len(attack_name))
    fig.set_size_inches((SCALE * 7.5, SCALE * 3.5))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    for row in rows:
        for column in range(len(attack_name)):
            if row == rows[0]:
                axs[row][column].set_title(attack_name[column], fontsize=FONT_SIZE)
            axs[row][column].grid('True')
            axs[row][column].tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
            if column == 0:
                axs[row][column].set_ylabel(y_label_list[row], fontsize=FONT_SIZE)
            if row == rows[-1]:
                axs[row][column].set_xlabel('Iteration', fontsize=FONT_SIZE)
                if plott == 'accregret':
                    axs[row][column].set_ylim(0, 100000)
                    axs[row][column].ticklabel_format(style='sci', scilimits=(-1, 1), axis='y')
                    axs[row][column].yaxis.get_offset_text().set_fontsize(LABEL_SIZE)
                else:
                    axs[row][column].set_ylim(0, 50)
            if row != rows[-1]:
                axs[row][column].set_xticklabels([])
                axs[row][column].set_ylim(0, 1)
            if column != 0:
                axs[row][column].set_yticklabels([])
            for i in range(len(algorithms)):
                axs[row][column].plot(iter[::100], plot_list[row][i][column][::100], label=labels[i], marker=markers[i],
                                      markevery=slice(0 + 1 * i, 100, 10), markersize=6,
                                      linestyle=linestyles[i])  # color=colors[i],

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    plt.subplots_adjust(bottom=0.2)
    fig.legend(lines, labels, loc='lower center', ncol=int(len(labels) / 2), fontsize=LEGEND_SIZE,
               bbox_to_anchor=(-0.051, -0.005, 1, 0.7))
    # fig.legend(lines, labels, loc='lower center', ncol=len(labels),fontsize=FONT_SIZE-3)
    #axs[0][-1].legend(lines, labels, loc='lower right', ncol=1, fontsize=15)

    save_path = '..\\picture\\{0}\\{1}\\{2}\\{3}\\setting-{4}-eta-{5}-momentum{6}.png'.format(plott, groot, descent_way,
                                                                    mode, setting, str(int(eta*10)), str(int(m*10)))
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    # plt.show()
    plt.close()



if __name__ == '__main__':
    eta_ls = [0.1, 2, 10]
    setting_ls = ['iid', 'noniid']
    mode_ls = ['default']
    plott_ls = ['accregret']  # 'avgregret'
    momentum_ls = [0.1, 1, 2, 10]
    for eta in eta_ls:
        for setting in setting_ls:
            for mode in mode_ls:
                for m in momentum_ls:
                    for plott in plott_ls:
                        draw_all(plott=plott, m=m,  setting=setting, eta=eta, mode=mode, groot='test', draw_iter=10000)
