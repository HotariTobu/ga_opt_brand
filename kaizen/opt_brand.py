from array import array
import csv
import random
import copy
import math
from statistics import mean

random.seed(0)

STOCK_LIST_FILEPATH = "stock.txt"
STOCK_DATA_FILEPATH_TEMPLATE = "{}.csv"

CLOSING_COL_INDEX = 4
"""CSVデータファイルの終値の列番号"""

POPULATION = 50
"""一世代の個体数"""

LOCUS_NUM = 4
"""遺伝子の数"""

MAXIMUM_TERMINAL = 500
"""最大世代数"""

MUTATION_RATE = 0.05
"""突然変異を起こす確率"""

stock_codes: list[str] = []
"""銘柄コードの一覧"""

roi_array_dict: dict[int, array[float]] = {}
"""
各銘柄の収益率の配列の辞書
収益率の配列は日付の降順となっている
"""

def closing_to_roi(closing_array: array[float]) -> array[float]:
    """終値の配列から収益率の配列を作成する。

    Args:
        closing_array (array[float]): 終値の配列。日付の降順である必要がある。

    Returns:
        array[float]: 収益率の配列。終値の配列の大きさ - 1の大きさになる
    """

    len_roi_array = len(closing_array) - 1
    roi_array = array('f', [0] * len_roi_array)

    for i in range(len_roi_array):
        roi_array[i] = (closing_array[i] - closing_array[i + 1]) / closing_array[i + 1]

    return roi_array

with open(STOCK_LIST_FILEPATH, "r") as file:
    for i, line in enumerate(file):
        stock_code = line.strip()
        stock_codes.append(stock_code)

        with open(STOCK_DATA_FILEPATH_TEMPLATE.format(stock_code), "r") as csv_file:
            csv_file.readline()
            csv_reader = csv.reader(csv_file)
            closing_array = array('f', [float(row[CLOSING_COL_INDEX]) for row in csv_reader])
            roi_array_dict[i] = closing_to_roi(closing_array)

day_count = len(roi_array_dict[0])
"""収益率が計算できた日の数"""

average_roi_dict: dict[int, float] = {}
"""平均収益率のキャッシュ"""

cov_roi_dict: dict[frozenset, float] = {}
"""収益率の共分散のキャッシュ"""

#平均収益率を計算する関数
def get_average_roi(i: int) -> float:
    """収益率の平均を計算する。計算結果を`average_roi_dict`に格納し、2回目以降の呼び出しではその値を返す。

    Args:
        i (int): 銘柄番号

    Returns:
        float: 指定された銘柄の収益率の平均
    """

    global roi_array_dict, average_roi_dict

    if i in average_roi_dict:
        return average_roi_dict[i]

    roi_array = roi_array_dict[i]
    average_roi = mean(roi_array)
    average_roi_dict[i] = average_roi
    return average_roi

#共分散を計算する関数
def get_cov_roi(i: int, j: int) -> float:
    """収益率の共分散を計算する。計算結果を`cov_roi_dict`に格納し、2回目以降の呼び出しではその値を返す。

    Args:
        i (int): 銘柄番号1
        j (int): 銘柄番号2

    Returns:
        float: 指定された2つの銘柄の収益率の共分散
    """

    global roi_array_dict, day_count, cov_roi_dict

    key = frozenset({i, j})
    if key in cov_roi_dict:
        return cov_roi_dict[key]

    # cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]

    roi_array_i = roi_array_dict[i]
    roi_array_j = roi_array_dict[j]
    sum_roi_ij = 0
    for roi_i, roi_j in zip(roi_array_i, roi_array_j):
        sum_roi_ij += roi_i * roi_j
    average_roi_ij = sum_roi_ij / day_count

    average_roi_i = get_average_roi(i)
    average_roi_j = get_average_roi(j)

    cov_roi = average_roi_ij - average_roi_i * average_roi_j

    cov_roi_dict[key] = cov_roi

    return cov_roi

investment_ratio = 1 / LOCUS_NUM
"""投資比率(固定。個体内で均等)"""

def calc_risks(individuals: list[list[int]]) -> array[float]:
    global investment_ratio

    risk_array = array('f')

    for individual in individuals:
        risk = 0

        for i in individual:
            for j in individual:
                risk += get_cov_roi(i, j) * investment_ratio * investment_ratio

        risk_array.append(risk)

    return risk_array

def calc_returns(individuals: list[list[int]]) -> array[float]:
    global investment_ratio

    return_array = array('f')

    for individual in individuals:
        ret = 0

        for i in individual:
            ret += get_average_roi(i) * investment_ratio

        return_array.append(ret)

    return return_array

individuals: list[list[int]] = []
"""
個体群

`[[4, 1, 2], [3, 5, 2], ...]`というように銘柄のインデックスのリストが1つの個体を表す。
"""

population_range = range(POPULATION)
"""個体数の範囲"""

stock_range = range(len(stock_codes))
"""銘柄番号の範囲"""

# 初期個体を発生させる
for _ in population_range:
    individual = random.sample(stock_range, LOCUS_NUM)
    individuals.append(individual)

def print_risk_return(individuals: list[list[int]], prefix = ''):
    """リスクとリターンを出力する。

    Args:
        individuals (list[list[int]]): 個体群
        prefix (str, optional): 行頭のプレフィクス
    """

    risk_array = calc_risks(individuals)
    return_array = calc_returns(individuals)
    print(prefix + 'risk_array =', risk_array)
    print(prefix + 'return_array =', return_array)

print("初期個体群 =", individuals)
print_risk_return(individuals, prefix="initial_")


def crossover(individuals: list[list[int]]):
    for _ in population_range:
        while True:
            #交叉する個体を示す変数
            i, j = random.sample(population_range, 2)
            individual_i = individuals[i]
            individual_j = individuals[j]

            #交叉する遺伝子座を選択
            locus = random.randrange(1, LOCUS_NUM)

            #交叉を行う
            new_individual_i = individual_i[:locus] + individual_j[locus:]
            new_individual_j = individual_j[:locus] + individual_i[locus:]

            #交叉する際, 一つの個体内に同じ銘柄番号が格納されていないか確認する
            if len(new_individual_i) == len(set(new_individual_i)) and len(new_individual_j) == len(set(new_individual_j)):
                individuals[i] = new_individual_i
                individuals[j] = new_individual_j
                break

def mutation(individuals: list[list[int]]):
    for i in population_range:
        if random.random() > MUTATION_RATE:
            continue

        new_individual = individuals[i].copy()

        #ランダムに銘柄番号を決める 同じ銘柄番号の場合はやり直し
        while True:
            #変更前と変更後の銘柄が同じじゃない場合銘柄を変更
            stock_index = random.choice(stock_range)
            if stock_index not in new_individual:
                break

        #どの遺伝子座に突然変異を施すか決める
        locus = random.randrange(LOCUS_NUM)

        new_individual[locus] = stock_index
        individuals[i] = new_individual

#適合度計算関数
def calc_gofs(risk_array: array[float], return_array: array[float]) -> array[int]:
    len_gof_array = len(risk_array)
    gof_array = array('i', [0] * len_gof_array)

    for i in range(len_gof_array):
        for j in range(i + 1, len_gof_array):
            risk_i = risk_array[i]
            risk_j = risk_array[j]
            ret_i = return_array[i]
            ret_j = return_array[j]

            #個体iが個体jよりリスクが高く, リターンが低い場合
            if (risk_i >= risk_j) and (ret_i <= ret_j):
                gof_array[i] += 1

            #個体jが個体iよりリスクが高く, リターンが低い場合
            if (risk_i <= risk_j) and (ret_i >= ret_j):
                gof_array[j] += 1

    return gof_array

for terminal in range(MAXIMUM_TERMINAL):
    #Individuals replication
    #一世代前の個体群をコピーして保存する
    prev_individuals = individuals.copy()

    crossover(individuals)
    mutation(individuals)

    #一世代前と現代の世代を一つのリストに入れる
    individuals += prev_individuals

    #Evaluation
    risk_array = calc_risks(individuals)
    return_array = calc_returns(individuals)
    #Evaluation終了

    #Environmental selection

    #適合度を計算する
    gof_array = calc_gofs(risk_array, return_array)

    #次世代に残す個体をリストで保存
    next_individuals: list[list[int]] = []

    #適合度0の個体数が次世代に残す所定個体数(POPULATION)以下であるかどうか判定
    if gof_array.count(0) <= POPULATION:
        #適合度が小さい順、リスクが小さい順に個体を選ぶ
        next_individuals = [individual for _, _, individual in sorted(zip(gof_array, risk_array, individuals))[:POPULATION]]
    #適合度0の個体が次世代に残す所定個体数以下である場合の処理終了
    else:
        raise NotImplementedError()
        # #適合度0の個体が次世代に残す個体より多い場合
        # #個体度0の個体のindividualsのインデックスをリストに格納
        # next_idx: list[int] = []
        # for i in range(len(gof_array)):
        #     if gof_array[i] == 0:
        #         next_idx.append(i)

        # #next_idxの要素数がpopulationと等しくなるまで繰り返す
        # for _ in range(POPULATION, len(next_idx)):
        #     #next_idxのすべての組み合わせに対して距離を保存するリスト
        #     #リストの形式は[(距離, 個体のインデックス1, 個体のインデックス2), ...]
        #     next_idx_distance: list[tuple[float, int, int]] = []
        #     #Step 1. next_idxのすべての組み合わせに対して距離を計算, リストに保存
        #     for i, j in combinations(next_idx, 2):
        #         next_idx_distance.append((calculate_distance(p, i, j), i, j))
        #     #Step2. もっとも距離の短い2点を見つけ出しmin_distanceにタプル形式で代入する
        #     min_distance = min(next_idx_distance, key=lambda x: x[0])
        #     #Step3. もっとも近接している2個体それぞれの個体に対して, 2番目に近接している個体との距離を計算
        #     min_dis1: list[tuple[float, int, int]] = []
        #     min_dis2: list[tuple[float, int, int]] = []
        #     for i in range(len(next_idx)):
        #         if i == min_distance[1] or i == min_distance[2]:
        #             continue

        #         min_dis1.append((calculate_distance(p, min_distance[1], i), min_distance[1], i))
        #         min_dis2.append((calculate_distance(p, min_distance[2], i), min_distance[2], i))

        #     min_dis1_min = min(min_dis1, key=lambda x: x[0])
        #     min_dis2_min = min(min_dis2, key=lambda x: x[0])
        #     #Step4. 2番目に近接している個体との距離が短い個体を削除する
        #     if min_dis1_min[0] < min_dis2_min[0]:
        #         #min_dis1_minのインデックスをnext_idxから削除
        #         next_idx.remove(min_dis1_min[1])
        #     else:
        #         #min_dis2_minのインデックスをnext_idxから削除
        #         next_idx.remove(min_dis2_min[1])
        # #next_idxの要素数がpopulationと等しくなるまでループ完了

        # for row in next_idx:
        #     next_individuals.append(individuals[row])
    #次世代に残す個体の選定終了
    individuals = next_individuals
    print(terminal, end='\r')
#一世代での操作終了

print("最適化された銘柄の組み合わせ =", individuals)
print_risk_return(individuals, prefix="optimized_")
