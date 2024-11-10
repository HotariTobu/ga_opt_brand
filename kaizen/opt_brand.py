from array import array
import csv
import random
import math
from statistics import mean

random.seed(0)

STOCK_LIST_FILEPATH = "stock.txt"
"""銘柄リストのファイルのパス"""

STOCK_DATA_FILEPATH_TEMPLATE = "{}.csv"
"""銘柄コードを埋め込んで株式データのファイルのパスを得るためのテンプレート"""

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
収益率の配列の要素は日付の降順となっている
"""


def closing_to_roi(closing_array: array[float]) -> array[float]:
    """終値の配列から収益率の配列を作成する。

    Args:
        closing_array (array[float]): 終値の配列。日付の降順である必要がある。

    Returns:
        array[float]: 収益率の配列。終値の配列の大きさ - 1の大きさになる
    """

    len_roi_array = len(closing_array) - 1
    roi_array = array("f", [0] * len_roi_array)

    for i in range(len_roi_array):
        roi_array[i] = (closing_array[i] - closing_array[i + 1]) / closing_array[i + 1]

    return roi_array


# 株式データのファイルを読み込む
with open(STOCK_LIST_FILEPATH, "r") as file:
    for i, line in enumerate(file):
        stock_code = line.strip()
        stock_codes.append(stock_code)

        with open(STOCK_DATA_FILEPATH_TEMPLATE.format(stock_code), "r") as csv_file:
            csv_file.readline()
            csv_reader = csv.reader(csv_file)

            closing_list = [float(row[CLOSING_COL_INDEX]) for row in csv_reader]
            closing_array = array("f", closing_list)

            roi_array_dict[i] = closing_to_roi(closing_array)

day_count = len(roi_array_dict[0])
"""収益率が計算できた日の数"""

average_roi_dict: dict[int, float] = {}
"""平均収益率のキャッシュ"""

cov_roi_dict: dict[frozenset, float] = {}
"""収益率の共分散のキャッシュ"""


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

    # E[XY]の計算
    roi_array_i = roi_array_dict[i]
    roi_array_j = roi_array_dict[j]
    sum_roi_ij = 0
    for roi_i, roi_j in zip(roi_array_i, roi_array_j):
        sum_roi_ij += roi_i * roi_j
    average_roi_ij = sum_roi_ij / day_count

    # E[X]とE[Y]の取得
    average_roi_i = get_average_roi(i)
    average_roi_j = get_average_roi(j)

    cov_roi = average_roi_ij - average_roi_i * average_roi_j

    cov_roi_dict[key] = cov_roi

    return cov_roi


investment_ratio = 1 / LOCUS_NUM
"""投資比率(固定。個体内で均等)"""


def calc_risks(individuals: list[list[int]]) -> array[float]:
    """指定された個体群のリスクを計算する。

    Args:
        individuals (list[list[int]]): 個体群

    Returns:
        array[float]: 個体ごとのリスク
    """

    global investment_ratio

    risk_array = array("f")

    for individual in individuals:
        risk = 0

        for i in individual:
            for j in individual:
                risk += get_cov_roi(i, j) * investment_ratio * investment_ratio

        risk_array.append(risk)

    return risk_array


def calc_returns(individuals: list[list[int]]) -> array[float]:
    """指定された個体群のリターンを計算する。

    Args:
        individuals (list[list[int]]): 個体群

    Returns:
        array[float]: 個体ごとのリターン
    """

    global investment_ratio

    return_array = array("f")

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


def print_stock_combinations(individuals: list[list[int]], prefix=""):
    """銘柄の組み合わせを出力する。

    Args:
        individuals (list[list[int]]): 個体群
        prefix (str, optional): 行頭のプレフィクス
    """

    combinations = [stock_codes[i] for individual in individuals for i in individual]
    print(prefix + "combinations =", combinations)


def print_risk_return(individuals: list[list[int]], prefix=""):
    """リスクとリターンを出力する。

    Args:
        individuals (list[list[int]]): 個体群
        prefix (str, optional): 行頭のプレフィクス
    """

    risk_array = calc_risks(individuals)
    return_array = calc_returns(individuals)
    print(prefix + "risk_array =", risk_array)
    print(prefix + "return_array =", return_array)


print_stock_combinations(individuals, prefix="initial_")
print_risk_return(individuals, prefix="initial_")


def crossover(individuals: list[list[int]]):
    """交叉を行って個体群を更新する。

    Args:
        individuals (list[list[int]]): 個体群
    """

    for _ in population_range:
        while True:
            # 交叉する個体を選ぶ
            i, j = random.sample(population_range, 2)
            individual_i = individuals[i]
            individual_j = individuals[j]

            # 交叉する遺伝子座を選ぶ
            locus = random.randrange(1, LOCUS_NUM)

            # 交叉を行う
            new_i = individual_i[:locus] + individual_j[locus:]
            new_j = individual_j[:locus] + individual_i[locus:]

            # 交叉後、一つの個体に同じ銘柄番号が含まれる場合はやり直し
            if len(new_i) == len(set(new_i)) and len(new_j) == len(set(new_j)):
                individuals[i] = new_i
                individuals[j] = new_j
                break


def mutation(individuals: list[list[int]]):
    """突然変異を行って個体群を更新する。

    Args:
        individuals (list[list[int]]): 個体群
    """

    for i in population_range:
        if random.random() > MUTATION_RATE:
            continue

        new = individuals[i].copy()

        # 元の個体に含まれない銘柄番号を選ぶ
        while True:
            stock_index = random.choice(stock_range)
            if stock_index not in new:
                break

        # 突然変異する遺伝子座を選ぶ
        locus = random.randrange(LOCUS_NUM)

        new[locus] = stock_index
        individuals[i] = new


def calc_gofs(risk_array: array[float], return_array: array[float]) -> list[int]:
    """リスクとリターンから適合度を計算する。

    Args:
        risk_array (array[float]): リスクの配列
        return_array (array[float]): リターンの配列

    Returns:
        list[int]: 適合度のリスト
    """

    # リスクの低い順、リターンの高い順で個体のインデックスを並び替える
    sorted_risk_indices = [
        i for i, _ in sorted(enumerate(risk_array), key=lambda x: x[1])
    ]
    sorted_return_indices = [
        i for i, _ in sorted(enumerate(return_array), key=lambda x: -x[1])
    ]

    # 個体のインデックスからリスクの順位、リターンの順位を取得するための辞書
    risk_order_dict = {i: order for order, i in enumerate(sorted_risk_indices)}
    return_order_dict = {i: order for order, i in enumerate(sorted_return_indices)}

    gof_list: list[int] = []

    for i in range(len(risk_array)):
        # 個体のリスクの順位とリターンの順位を取得する
        risk_order = risk_order_dict[i]
        return_order = return_order_dict[i]

        # この個体よりもリスクが低い個体のインデックスと、この個体よりもリターンが高い個体のインデックスを取得する
        low_risk_indices = sorted_risk_indices[:risk_order]
        high_return_indices = sorted_return_indices[:return_order]

        # リスクのインデックスとリターンのインデックスで重複している個体の数が適合度となる
        gof = len(set(low_risk_indices) & set(high_return_indices))
        gof_list.append(gof)

    return gof_list


# 進化開始
for terminal in range(MAXIMUM_TERMINAL):
    # Individuals replication
    # 一世代前の個体群をコピーして保存する
    prev_individuals = individuals.copy()

    crossover(individuals)
    mutation(individuals)

    # 一世代前と現代の世代を一つのリストに入れる
    individuals += prev_individuals

    # Evaluation
    risk_array = calc_risks(individuals)
    return_array = calc_returns(individuals)

    # Environmental selection

    # 適合度を計算する
    gof_list = calc_gofs(risk_array, return_array)

    # 次世代に残す個体のリスト
    next_individuals: list[list[int]] = []

    # 適合度0の個体数が次世代に残す所定個体数(POPULATION)以下であるかどうか判定
    if gof_list.count(0) <= POPULATION:
        # 適合度が小さい順、リスクが小さい順に個体を選ぶ
        next_individuals = [
            individual
            for _, _, individual in sorted(zip(gof_list, risk_array, individuals))[
                :POPULATION
            ]
        ]

    else:
        # 50000世代まで行っても発生しなかった_(:3」∠)_
        raise NotImplementedError()
        # #適合度0の個体が次世代に残す個体より多い場合
        # #個体度0の個体のindividualsのインデックスをリストに格納
        # next_idx: list[int] = []
        # for i in range(len(gof_list)):
        #     if gof_list[i] == 0:
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

    individuals = next_individuals
    print(terminal, end="\r")

print_stock_combinations(individuals, prefix="optimized_")
print_risk_return(individuals, prefix="optimized_")
