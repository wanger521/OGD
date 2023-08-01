import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config


def draw_all(plott='accregret', descent_way='model', m=0, eta=0.1, setting='iid', mode='default', groot = 'artificial',
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

    last_str = ['', '-sf', '-gs']
    attack_name = ['without attack', 'sign flipping attack', 'Gaussian attack']
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


    loss_list = []
    regret_list = []
    sregret_list = []
    FONT_SIZE = 24
    LABEL_SIZE = 15
    SCALE = 2.5
    for i in range(len(algorithms)):
        loss_alo = []
        sregret_alo = []
        regret_alo = []
        for j in range(len(last_str)):
            path_open_iid = '..\\results\\{0}\\{1}\\{2}\\{2}{3}-step{4}-setting-{5}-momentum{6}.pkl'.format(groot,
                            descent_way, algorithms[i], last_str[j], str(int(eta*10)), 'iid', str(int(m*10)))
            path_open_noniid = '..\\results\\{0}\\{1}\\{2}\\{2}{3}-step{4}-setting-{5}-momentum{6}.pkl'.format(groot,
                            descent_way, algorithms[i], last_str[j], str(int(eta*10)), 'noniid', str(int(m*10)))
            with open(path_open_iid, 'rb') as f:
                _, sregret, _ = pickle.load(f)
                sregret_alo.append(sregret[:draw_iter])
            with open(path_open_noniid, 'rb') as f:
                _, _, regret = pickle.load(f)
                regret_alo.append(regret[:draw_iter])
        sregret_list.append(sregret_alo)
        regret_list.append(regret_alo)

    # Plot the curve
    rows = []
    if plott == 'regret':
        rows = [0, 1]
        plot_list = [loss_list] + [regret_list]
        y_label_list = ['Loss', 'Adversarial Regret']
    elif plott  == 'sregret':
        rows = [0, 1]
        plot_list = [loss_list] + [sregret_list]
        y_label_list = ['Loss', 'Stochastic Regret']
    elif plott == 'all':
        rows = [0, 1]
        plot_list = [sregret_list] + [regret_list]
        y_label_list = ['Stochastic Regret', 'Adversarial Regret']

    fig, axs = plt.subplots(len(rows), len(attack_name))
    fig.set_size_inches((SCALE * 8, SCALE * 3.5))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    for row in rows:
        for column in range(len(attack_name)):
            axs[row][column].set_title(attack_name[column], fontsize=FONT_SIZE)
            axs[row][column].grid('True')
            axs[row][column].tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
            if column == 0:
                axs[row][column].set_ylabel(y_label_list[row], fontsize=FONT_SIZE - 1)
            if row == rows[-1]:
                axs[row][column].set_xlabel('Iteration', fontsize=FONT_SIZE)
            if row != rows[0]:
                axs[row][column].set_ylim(0, 12000)
                # axs[row][column].ticklabel_format(style='sci', scilimits=(-1, 1), axis='y')
                # axs[row][column].yaxis.get_offset_text().set_fontsize(LABEL_SIZE)
            if row != rows[-1]:
                axs[row][column].set_xticklabels([])
                # axs[row][column].set_ylim(0, 1)
            if column != 0:
                axs[row][column].set_yticklabels([])
            for i in range(len(algorithms)):
                axs[row][column].plot(iter, plot_list[row][i][column], label=labels[i], marker=markers[i],
                                      markevery=slice(0 + 100 * i, 10000, 1000), markersize=8,
                                      linestyle=linestyles[i])  # color=colors[i],

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='lower center', ncol=len(labels),fontsize=FONT_SIZE-3)
    axs[0][-1].legend(lines, labels, loc='lower right', ncol=1, fontsize=15)

    save_path = '..\\picture\\{0}\\{1}\\{2}\\{3}\\eta-{4}-momentum{5}.png'.format(plott, groot, descent_way,
                                                                    mode, str(int(eta*10)), str(int(m*10)))
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()



if __name__ == '__main__':
    eta_ls = [0.005, 0.1, 4]
    setting_ls = ['noniid', 'iid']#'iid',
    mode_ls = ['default']
    plott_ls = ['all']  # 'avgregret'
    momentum_ls = [0.005, 0.1, 4]
    for eta in eta_ls:
        for setting in setting_ls:
            for mode in mode_ls:
                for m in momentum_ls:
                    for plott in plott_ls:
                        draw_all(plott=plott, m=m,  setting=setting, eta=eta, mode=mode, groot='artificial2', draw_iter=5000)
