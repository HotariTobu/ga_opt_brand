from array import array
from collections import namedtuple
import csv
from itertools import combinations
import random
import copy
import math
from statistics import mean

random.seed(0)

CLOSING_COL_INDEX = 4
"""CSVデータファイルの終値の列番号"""

POPULATION = 50
"""一世代の個体数"""

LOCUS = 4
"""遺伝子座の数"""

MAXIMUM_TERMINAL = 500
"""最大世代数"""

PopulationItem = namedtuple('PopulationItem', ['risk', 'ret'])

investment_ratio = 1 / LOCUS
"""投資比率(固定。個体内で均等)"""

average_roi_dict: dict[int, float] = {}
"""平均収益率のキャッシュ"""

cov_roi_dict: dict[frozenset, float] = {}
"""収益率の共分散のキャッシュ"""

def calc_risk_return(individuals: list[list[int]]):
    #リターンとリスクを保存するリスト, 行は個体に対応
    returns = []
    risks = []

    #リターンを計算 投資比率は1/locusで固定(な投資比率)
    for individual in individuals:
        ret = 0
        #一つの個体のリターンを計算
        for i in individual:
            ret += r(i) * investment_ratio
        returns.append(ret)
    #リターンの計算終了

    #リスクを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
    for individual in individuals:
        ris = 0
        for i in individual:
            for j in individual:
                ris += sigma(i, j) * investment_ratio * investment_ratio
        risks.append(ris)
    #リスクの計算終了

    #[(risk, return), (risk, return), ...]に形を整える
    p = [PopulationItem(x, y) for x, y in zip(risks, returns)]

    return p

#平均収益率を計算する関数
def r(i):
    if i in average_roi_dict:
        return average_roi_dict[i]

    roi_list = roi_list_dict[i]
    average_roi = mean(roi_list)
    average_roi_dict[i] = average_roi
    return average_roi

#共分散を計算する関数
def sigma(i, j):
    key = frozenset({i, j})
    if key in cov_roi_dict:
        return cov_roi_dict[key]

    # cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]

    roi_list_i = roi_list_dict[i]
    roi_list_j = roi_list_dict[j]
    sum_roi_ij = 0
    for roi_i, roi_j in zip(roi_list_i, roi_list_j):
        sum_roi_ij += roi_i * roi_j
    average_roi_ij = sum_roi_ij / day_count

    average_roi_i = r(i)
    average_roi_j = r(j)

    cov_roi = average_roi_ij - average_roi_i * average_roi_j

    cov_roi_dict[key] = cov_roi

    return cov_roi

#適合度計算関数
def precision(population: list[PopulationItem]):
    #計算した適合度を格納しておくリスト
    precision_ans =[0] * len(population)
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            risk_i, ret_i = population[i]
            risk_j, ret_j = population[j]

            #個体iが個体jよりリスクが高く, リターンが低い場合
            if (risk_i >= risk_j) and (ret_i <= ret_j):
                precision_ans[i] += 1

            #個体jが個体iよりリスクが高く, リターンが低い場合
            if (risk_i <= risk_j) and (ret_i >= ret_j):
                precision_ans[j] += 1

    return precision_ans
#適合度計算関数終了

#2点間の距離を計算する関数
def calculate_distance(p: list[PopulationItem], idx_1, idx_2):
    x1, y1 = p[idx_1]
    x2, y2 = p[idx_2]

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
#2点間の距離の計算終了


#main文

#Readind data.

type StockCode = str
"""銘柄コード"""

stock_codes: list[StockCode] = []
"""銘柄コードの一覧"""

roi_list_dict: dict[int, array[float]] = {}
"""
各銘柄の収益率のリストの辞書
収益率のリストは日付の降順となっている
"""

roi_list_dict: dict[int, array[float]] = {}

def closing_to_roi(closing_list: array[float]) -> array[float]:
    """終値のリストから収益率のリストを作成する。

    Args:
        closing_list (array[float]): 終値のリスト。日付の降順である必要がある。

    Returns:
        array[float]: 収益率のリスト。終値のリストの大きさ - 1の大きさになる
    """

    len_roi_list = len(closing_list) - 1
    roi_list = array('f', [0] * len_roi_list)

    for i in range(len_roi_list):
        roi_list[i] = (closing_list[i] - closing_list[i + 1]) / closing_list[i + 1]

    return roi_list

with open("stock.txt", "r") as file:
    for i, line in enumerate(file):
        stock_code = line.strip()
        stock_codes.append(stock_code)

        with open(stock_code + ".csv", "r", encoding = "utf-8") as csv_file:
            csv_file.readline()
            csv_reader = csv.reader(csv_file)
            closing_list = array('f', [float(row[CLOSING_COL_INDEX]) for row in csv_reader])
            roi_list_dict[i] = closing_to_roi(closing_list)

day_count = len(roi_list_dict[0])
"""収益率が計算できた日の数"""

#Gen Initial individuals.

#初期個体群
initialIndividuals: list[list[int]] = []

population_range = range(POPULATION)

stock_range = range(len(stock_codes))
for _ in population_range:
    # 初期個体を発生させる
    individual = random.sample(stock_range, LOCUS)
    initialIndividuals.append(individual)

print("初期個体群 = ")
print(initialIndividuals)
print("初期個体群のリスクとリターン = ")
print(calc_risk_return(initialIndividuals))


#多目的GA

#現在の個体群を保存するリスト, [['1', '2', '3'], ['3', '5', '6']]のように管理されている
individuals: list[list[int]] = initialIndividuals

for terminal in range(MAXIMUM_TERMINAL):
    #Individuals replication
    #一世代前の個体群をコピーして保存する
    priorIndividuals = copy.deepcopy(individuals)
    #Crossover
    for _ in population_range:
        while True:
            #交叉する個体を示す変数
            I1 = random.randint(0, POPULATION - 1)
            I2 = random.randint(I1 + 1, I1 + POPULATION - 1) % POPULATION

            #交叉する遺伝子座を選択
            cLocus = random.randint(1, LOCUS - 1)

            #交叉を行う
            child1 = individuals[I1][:cLocus] + individuals[I2][cLocus:]
            child2 = individuals[I2][:cLocus] + individuals[I1][cLocus:]

            #交叉する際, 一つの個体内に同じ銘柄番号が格納されていないか確認する
            if (len(child1) == len(set(child1))) and  (len(child2) == len(set(child2))):
                individuals[I1] = child1
                individuals[I2] = child2
                break
    #Crossover終了

    #Mutation
    for i in population_range:
        #各個体に対して5%の確率で突然変異を施す
        if random.random() > 0.05:
            continue

        #どの遺伝子座に突然変異を施すか決める
        mutationLocus = random.randrange(LOCUS)
        #ランダムに銘柄番号を決める 同じ銘柄番号の場合はやり直し
        while True:
            child = individuals[i]
            #変更前と変更後の銘柄が同じじゃない場合銘柄を変更
            while True:
                tmp = random.choice(stock_range)
                if tmp != child[mutationLocus]:
                    child[mutationLocus] = tmp
                    break
            #遺伝子座の銘柄を変更完了

            if len(child) == len(set(child)):
                individuals[i] = child
                break
        #ランダムに銘柄番号決定終了
    #Mutation終了

    #一世代前と現代の世代を一つのリストに入れる
    individuals += priorIndividuals

    #Evaluation
    p = calc_risk_return(individuals)
    #Evaluation終了

    #Environmental selection

    #適合度を計算する
    precision_list = precision(p)

    #次世代に残す個体をリストで保存
    nextIndividuals: list[list[int]] = []

    #適合度0の個体数が次世代に残す所定個体数(POPULATION)以下であるかどうか判定
    if precision_list.count(0) <= POPULATION:
        #適合度が小さい順、リスクが小さい順に個体を選ぶ
        nextIndividuals = [individual for _, _, individual in sorted(zip(precision_list, p, individuals), key=lambda x: x)[:POPULATION]]
    #適合度0の個体が次世代に残す所定個体数以下である場合の処理終了
    else:
        raise NotImplementedError()
        #適合度0の個体が次世代に残す個体より多い場合
        #個体度0の個体のindividualsのインデックスをリストに格納
        next_idx: list[int] = []
        for i in range(len(precision_list)):
            if precision_list[i] == 0:
                next_idx.append(i)

        #next_idxの要素数がpopulationと等しくなるまで繰り返す
        for _ in range(POPULATION, len(next_idx)):
            #next_idxのすべての組み合わせに対して距離を保存するリスト
            #リストの形式は[(距離, 個体のインデックス1, 個体のインデックス2), ...]
            next_idx_distance: list[tuple[float, int, int]] = []
            #Step 1. next_idxのすべての組み合わせに対して距離を計算, リストに保存
            for i, j in combinations(next_idx, 2):
                next_idx_distance.append((calculate_distance(p, i, j), i, j))
            #Step2. もっとも距離の短い2点を見つけ出しmin_distanceにタプル形式で代入する
            min_distance = min(next_idx_distance, key=lambda x: x[0])
            #Step3. もっとも近接している2個体それぞれの個体に対して, 2番目に近接している個体との距離を計算
            min_dis1: list[tuple[float, int, int]] = []
            min_dis2: list[tuple[float, int, int]] = []
            for i in range(len(next_idx)):
                if i == min_distance[1] or i == min_distance[2]:
                    continue

                min_dis1.append((calculate_distance(p, min_distance[1], i), min_distance[1], i))
                min_dis2.append((calculate_distance(p, min_distance[2], i), min_distance[2], i))

            min_dis1_min = min(min_dis1, key=lambda x: x[0])
            min_dis2_min = min(min_dis2, key=lambda x: x[0])
            #Step4. 2番目に近接している個体との距離が短い個体を削除する
            if min_dis1_min[0] < min_dis2_min[0]:
                #min_dis1_minのインデックスをnext_idxから削除
                next_idx.remove(min_dis1_min[1])
            else:
                #min_dis2_minのインデックスをnext_idxから削除
                next_idx.remove(min_dis2_min[1])
        #next_idxの要素数がpopulationと等しくなるまでループ完了

        for row in next_idx:
            nextIndividuals.append(individuals[row])
    #次世代に残す個体の選定終了
    individuals = nextIndividuals
    print(terminal, end='\r')
#一世代での操作終了

print("最適化された銘柄の組み合わせ = ")
print(individuals)
print("最適化された個体の個体数")
print(len(individuals))
p = calc_risk_return(individuals)
print("最適化された銘柄のリターンリスク = ")
print(p)
print("最適化されたリスクとリターンの個体数")
print(len(p))
