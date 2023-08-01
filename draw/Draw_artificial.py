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
    # print(eta)
    # print(m)
    # print(setting)

    iter = [i+1 for i in range(draw_iter)]

    last_str = ['', '-sf', '-gs', '-hd']
    attack_name = ['without attack', 'sign-flipping attack', 'Gaussian attack', 'sample-duplicating attack']
    # last_str = ['', '-sf', '-gs']
    # attack_name = ['without attack', 'sign flipping attack', 'Gaussian attack']
    # last_str = ['', '-sf']
    # attack_name = ['without attack', 'sign flipping attack']
    algorithms = []
    colors = []
    markers = []
    labels = []
    if mode == 'default' or mode=='rebuttal':
        algorithms = ['Mean', 'CMedian', 'trimmed-mean', 'GeoMed',  'Krum', 'CenterClip', 'Phocas', 'FABA']
        # colors = ['black','gold', 'skyblue', 'brown', 'olive', 'blue', 'darkgray', 'purple']
        markers = ['h', '1', 'v', 'o', 'x', '*', 'd', '+']
        linestyles = ['-', '-.', ':', '--', '-', '-.', ':', '--']
        labels = ['mean', 'coordinate-wise median', 'trimmed mean', 'geometric median', 'Krum', 'centered clipping', 'Phocas', 'FABA']


    loss_list = []
    regret_list = []
    sregret_list = []
    loss_list_original = []
    regret_list_original = []
    sregret_list_original = []
    FONT_SIZE = 20
    LEGEND_SIZE = 20
    LABEL_SIZE = 15
    SCALE = 2.5
    for i in range(len(algorithms)):
        loss_alo = []
        sregret_alo = []
        regret_alo = []
        loss_alo_original = []
        sregret_alo_original = []
        regret_alo_original = []
        for j in range(len(last_str)):
            path_open = '..\\results\\{0}\\{1}\\{2}\\{2}{3}-step{4}-setting-{5}-momentum{6}.pkl'.format(groot,
                            descent_way, algorithms[i], last_str[j], str(int(eta*10)), setting, str(int(m*10)))
            with open(path_open, 'rb') as f:
                loss, sregret, regret = pickle.load(f)
                # if plott =='avgregret':
                np_r = np.array(regret[:draw_iter])
                np_i = np.array(iter)
                #regret =list(np.divide(np_r, np_i))
                #sregret = list(np.divide(np.array(sregret[:draw_iter]), np_i))
                loss_alo.append(loss[:draw_iter])
                sregret_alo.append(sregret[:draw_iter])
                regret_alo.append(regret[:draw_iter])
            path_open_original = '..\\results\\{0}\\{1}\\{2}\\{2}{3}-step{4}-setting-{5}-momentum{6}.pkl'.format(groot,
                            descent_way, algorithms[i], last_str[j], str(int(eta*10)), setting, str(int(10)))
            with open(path_open_original, 'rb') as f:
                loss_original, sregret_original, regret_original = pickle.load(f)
                # if plott =='avgregret':
                np_r = np.array(regret_original[:draw_iter])
                np_i = np.array(iter)
                #regret =list(np.divide(np_r, np_i))
                #sregret = list(np.divide(np.array(sregret[:draw_iter]), np_i))
                loss_alo_original.append(loss_original[:draw_iter])
                sregret_alo_original.append(sregret_original[:draw_iter])
                regret_alo_original.append(regret_original[:draw_iter])
        loss_list.append(loss_alo)
        sregret_list.append(sregret_alo)
        regret_list.append(regret_alo)
        loss_list_original.append(loss_alo_original)
        sregret_list_original.append(sregret_alo_original)
        regret_list_original.append(regret_alo_original)

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
    elif plott == 'two':
        if len(attack_name) ==2:
            FONT_SIZE = 20#30
            LEGEND_SIZE = 20
            LABEL_SIZE = 20
        else:
            FONT_SIZE = 20#24
            LABEL_SIZE = 15
            LEGEND_SIZE = 20
            SCALE = 2.5
        if setting == 'iid':
            rows = [0, 1]
            plot_list = [regret_list] + [sregret_list]
            y_label_list = ['Adversarial Regret', 'Stochastic Regret']
        else:
            rows = [0]
            plot_list = [regret_list]
            y_label_list = ['Adversarial Regret']
    elif plott == 'all':
        rows = [0, 1, 2]
        plot_list = [loss_list] + [sregret_list] + [regret_list]
        y_label_list = ['Loss', 'Stochastic Regret', 'Adversarial Regret']
    elif plott == 'compare_momentum':
        if len(attack_name) ==2:
            FONT_SIZE = 30
            LEGEND_SIZE = 20
            LABEL_SIZE = 20
        else:
            FONT_SIZE = 24
            LABEL_SIZE = 15
            SCALE = 2.5
        if setting == 'iid':
            rows = [0, 1]
            plot_list = [sregret_list_original] + [sregret_list]
            y_label_list = ['Stochastic Regret', 'Stochastic Regret']
        else:
            rows = [0, 1]
            plot_list = [regret_list_original] + [regret_list]
            y_label_list = ['Stochastic Regret', 'Stochastic Regret']

    # fig, axs = plt.subplots(len(rows), len(attack_name))
    # if len(rows) == 1:
    #     fig.set_size_inches((SCALE * 8, SCALE * 1.5))
    # else:
    #     if len(attack_name)==2:
    #         fig.set_size_inches((SCALE * 8, SCALE * 4.5))
    #     else:
    #         fig.set_size_inches((SCALE * 8, SCALE * 3.5))
    # plt.subplots_adjust(hspace=0.2, wspace=0.1)
    fig, axs = plt.subplots(len(rows), len(attack_name))
    if len(rows) == 1:
        fig.set_size_inches((SCALE * 7.5, SCALE * 2))
    else:
        if len(attack_name) == 2:
            fig.set_size_inches((SCALE * 7.5, SCALE * 4.5))
        else:
            fig.set_size_inches((SCALE * 7.5, SCALE * 3.5))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    for row in rows:
        for column in range(len(attack_name)):
            if len(rows) == 1:
                sub = axs[column]
            else:
                sub = axs[row][column]
            if row == rows[0]:
                sub.set_title(attack_name[column], fontsize=FONT_SIZE)
            sub.grid('True')
            sub.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
            if column == 0:
                sub.set_ylabel(y_label_list[row], fontsize=FONT_SIZE )#-2
            if row == rows[-1]:
                sub.set_xlabel('Iteration', fontsize=FONT_SIZE)
            #if row != rows[0]:
            if setting == 'iid':
                sub.set_ylim(0, 40000)#20000
            else:
                if m == 1:
                    sub.set_ylim(0, 100000)
                else:
                    sub.set_ylim(0, 200000)
                    # sub.set_ylim(0, 500000)
            sub.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y', labelsize=LABEL_SIZE)
            sub.yaxis.get_offset_text().set_fontsize(LABEL_SIZE)
            if row != rows[-1]:
                sub.set_xticklabels([])
            #     # axs[row][column].set_ylim(0, 1)
            if column != 0:
                sub.set_yticklabels([])
            for i in range(len(algorithms)):
                sub.plot(iter, plot_list[row][i][column], label=labels[i], marker=markers[i],
                                      markevery=slice(0 + 100 * i, 10000, 400), markersize=6,
                                      linestyle=linestyles[i])  # color=colors[i],

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='lower center', ncol=len(labels),fontsize=FONT_SIZE-3)
    # if len(rows) == 1:
    #     axs[-1].legend(lines, labels, loc='lower right', ncol=1, fontsize=LEGEND_SIZE)
    # else:
    #     axs[0][-1].legend(lines, labels, loc='lower right', ncol=1, fontsize=LEGEND_SIZE)
    if len(rows) == 1:
        plt.subplots_adjust(bottom=0.35)
        fig.legend(lines, labels, loc='lower center', ncol=int(len(labels) / 2), fontsize=LEGEND_SIZE ,
                   bbox_to_anchor=(-0.053, -0.005, 1, 0.7))
    else:
        plt.subplots_adjust(bottom=0.2)
        fig.legend(lines, labels, loc='lower center', ncol=int(len(labels) / 2), fontsize=LEGEND_SIZE ,
                   bbox_to_anchor=(-0.051, -0.005, 1, 0.7))

    save_path = '..\\picture\\{0}\\{1}\\{2}\\{3}\\setting-{4}-eta-{5}-momentum{6}.png'.format(plott, groot, descent_way,
                                                                    mode, setting, str(int(eta*10)), str(int(m*10)))
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    #plt.show()
    plt.close()



if __name__ == '__main__':
    #eta_ls =[6,7]#[10,4] #[0.001, 4, 5]#[0.1, 0.005, 4]# [0.002]
    setting_ls =['noniid']#'iid',
    mode_ls = ['default']#['default']
    plott_ls = ['two']  # 'avgregret'
    #momentum_ls =[6,7,1]#[10,1,4]#[0.001, 1, 4, 5]#[0.1, 0.005,  4]#[0.1,0.01, 0.9, 1]
    eta_ls = [0.005, 4, 10]
    momentum_ls = [1, 0.005,  4, 10]
    #draw_all(plott='compare_momentum', m=10, setting='iid', eta=10, mode='rebuttal', groot='artificial3', draw_iter=5000)
    #draw_all(plott='two', m=10, setting='iid', eta=10, mode='rebuttal', groot='artificial3', draw_iter=5000)
    #draw_all(plott='two', m=1, setting='iid', eta=10, mode='rebuttal', groot='artificial3', draw_iter=5000)
    for eta in eta_ls:
        for setting in setting_ls:
            for mode in mode_ls:
                for m in [1,eta]:
                    for plott in plott_ls:
                        # print(eta)
                        # print(m)
                        draw_all(plott=plott, m=m,  setting=setting, eta=eta, mode=mode, groot='artificial5', draw_iter=2000)#之前在只有10个节点是是5000draw iter
