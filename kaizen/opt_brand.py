import csv
import random
import copy
import math

POPULATION = 50
"""一世代の個体数"""

LOCUS = 4
"""遺伝子座の数"""

MAXIMUM_TERMINAL = 500
"""最大世代数"""

#平均収益率を計算する関数
def r(stock_name, stock_dict, T):
    #各日の収益率を計算して足し合わせる
    erning_sum = 0
    for k in range(1, T):
        #当日の収益率
        erning = (float(stock_dict[stock_name][k][4]) - float(stock_dict[stock_name][k-1][4])) / float(stock_dict[stock_name][k-1][4])
        erning_sum = erning_sum + erning
    average = erning_sum / (T-1)
    return average

#共分散を計算する関数
def sigma(stock_namei, stock_namej, stock_dict, T, average_ri, average_rj):
    #偏差の合計を計算
    distinction_sum = 0
    for k in range(1, T):
        erning_i = (float(stock_dict[stock_namei][k][4]) - float(stock_dict[stock_namei][k-1][4])) / float(stock_dict[stock_namei][k-1][4])
        erning_j = (float(stock_dict[stock_namej][k][4]) - float(stock_dict[stock_namej][k-1][4])) / float(stock_dict[stock_namej][k-1][4])
        distinction = (erning_i - average_ri) * (erning_j - average_rj)
        distinction_sum = distinction_sum + distinction
    CoV = distinction_sum / (T-1)
    #print(CoV)
    return CoV

#適合度計算関数
def precision(population):
    #計算した適合度を格納しておくリスト
    precision_ans =[0] * len(population)
    for i in range(0, len(population)):
        for j in range(i + 1, len(population)):
            if i != j:
                #個体iが個体jよりリスクが低く, リターンが高い場合
                if (population[i][0] <= population[j][0]) and (population[i][1] >= population[j][1]):
                    precision_ans[j] = precision_ans[j] + 1

                #個体jが個体iよりリスクが低く, リターンが高い場合
                if (population[j][0] <= population[i][0]) and (population[j][1] >= population[i][1]):
                    precision_ans[i] = precision_ans[i] + 1

    return precision_ans
#適合度計算関数終了

#2点間の距離を計算する関数
def calculate_distance(p, idx_1, idx_2):
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

stock_dict: dict[StockCode, list[list[str]]] = {}
"""
株式データ[StockCode][Date][Column]

stock_dict[1234][2023-12-29][3]で銘柄コード1234の2023年12月29日の終値にアクセスできる。
"""

with open("stock.txt", "r") as file:
    for line in file:
        stock_code = line.strip()
        stock_codes.append(stock_code)

        with open(stock_code + ".csv", "r", encoding = "utf-8") as csv_file:
            f = list(csv.reader(csv_file, delimiter = ","))
            f.reverse()
            f.pop()
            stock_dict[stock_code] = [row for row in f]

#取得した株式データの期間を保存
T = len(stock_dict[stock_codes[0]])

#Gen Initial individuals.

#初期個体群
initialIndividuals = []

for _ in range(0,POPULATION):
    #重複が無いように初期個体を発生させる
    while True:
        tmp = []
        for i in range(0, LOCUS):
            tmp.append(stock_codes[random.randint(0, len(stock_codes) - 1)])
        if len(tmp) == len(set(tmp)):
            #発生させた個体を初期個体群にappend
            initialIndividuals.append(tmp)
            break
print("初期個体群 = ")
print(initialIndividuals)
print("初期個体群のリスクとリターン = ")
#Evaluation
#リターンとリスクを保存するリスト, 行は個体に対応
returns = []
risks = []

#リターンを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
for individual in initialIndividuals:
    ret = 0
    #一つの個体のリターンを計算
    for i in individual:
        ret = ret + (r(i, stock_dict, T) * (1/LOCUS))
    returns.append(ret)
#リターンの計算終了

#リスクを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
for individual in initialIndividuals:
    ris = 0
    for i in individual:
        for j in individual:
            ris = ris + (sigma(i, j, stock_dict, T, r(i, stock_dict, T), r(j, stock_dict, T)) * (1 / LOCUS) * (1 / LOCUS))
    risks.append(ris)
#リスクの計算終了

#[(risk, return), (risk, return), ...]に形を整える
p = [(x, y) for x, y in zip(risks, returns)]
#Evaluation終了
print(p)


#多目的GA

#現在の個体群を保存するリスト, [['1', '2', '3'], ['3', '5', '6']]のように管理されている
individuals = []
individuals = copy.deepcopy(initialIndividuals)

#1世代前の個体群を保存する
priorIndividuals = []

#世代数
terminal = 0

while True:
    #Terminal condition.
    if MAXIMUM_TERMINAL == terminal:
        #世代数が最大世代数なら終了
        break

    #Individuals replication
    #一世代前の個体群をコピーして保存する
    priorIndividuals = copy.deepcopy(individuals)
    #Crossover
    for i in range(0, POPULATION):
        while True:
            #交叉する個体を示す変数
            I1 = 0
            I2 = 0

            #交叉する個体を選定. 同一個体が選ばれている場合はやり直し
            while True:
                I1 = random.randint(0, POPULATION - 1)
                I2 = random.randint(0, POPULATION - 1)
                if I1 != I2:
                    break
            #交叉する個体の選定終了

            #交叉する遺伝子座を選択
            cLocus = 0
            cLocus = random.randint(1, LOCUS - 1)

            #交叉を行う
            child1 = individuals[I1][:cLocus] + individuals[I2][cLocus:]
            child2 = individuals[I2][:cLocus] + individuals[I1][cLocus:]

            I1tmp = []
            I2tmp = []
            I1tmp = copy.deepcopy(child1)
            I2tmp = copy.deepcopy(child2)
            I1tup = tuple()
            i2tup = tuple()
            I1tup = tuple(I1tmp)
            I2tup = tuple(I2tmp)

            #交叉する際, 一つの個体内に同じ銘柄番号が格納されていないか確認する
            if (len(I1tup) == len(set(I1tup))) and  (len(I2tup) == len(set(I2tup))):
                individuals[I1] = child1
                individuals[I2] = child2
                break
    #Crossover終了

    #Mutation
    for i in range(0, POPULATION):
        #各個体に対して5%の確率で突然変異を施す
        if random.random() < 0.05:
            #どの遺伝子座に突然変異を施すか決める
            mutationLocus = random.randint(0, LOCUS - 1)
            #ランダムに銘柄番号を決める 同じ銘柄番号の場合はやり直し
            while True:
                child = individuals[i]
                #変更前と変更後の銘柄が同じじゃない場合銘柄を変更
                while True:
                    tmp = stock_codes[random.randint(0, len(stock_codes) - 1)]
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
    individuals.extend(priorIndividuals)

    #Evaluation
    #リターンとリスクを保存するリスト, 行は個体に対応
    returns = []
    risks = []

    #リターンを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
    for individual in individuals:
        ret = 0
        #一つの個体のリターンを計算
        for i in individual:
            ret = ret + (r(i, stock_dict, T) * (1/LOCUS))
        returns.append(ret)
    #リターンの計算終了

    #リスクを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
    for individual in individuals:
        ris = 0
        for i in individual:
            for j in individual:
                ris = ris + (sigma(i, j, stock_dict, T, r(i, stock_dict, T), r(j, stock_dict, T)) * (1 / LOCUS) * (1 / LOCUS))
        risks.append(ris)
    #リスクの計算終了

    #[(risk, return), (risk, return), ...]に形を整える
    p = [(x, y) for x, y in zip(risks, returns)]
    #Evaluation終了

    #Environmental selection

    #適合度を計算する
    precision_list = precision(p)

    #次世代に残す個体をリストで保存
    nextIndividuals = []

    #適合度0の個体数が次世代に残す所定個体数(POPULATION)以下であるかどうか判定
    if precision_list.count(0) <= POPULATION:
        #適合度が小さい順にprecision_listのインデックスを並べる
        sorted_indices = [i for i, _ in sorted(enumerate(precision_list), key=lambda x: x[1])]
        if precision_list[sorted_indices[POPULATION]] != precision_list[sorted_indices[POPULATION - 1]]:
            #一世代分の個体を取る際, 切れ目が異なる適合度の場合の処理
            for i in range(POPULATION):
                nextIndividuals.append(individuals[sorted_indices[i]])
        else:
            #一世代分の個体を取る際, 切れ目が同じ適合度の場合の処理
            #同世代においてもリスクが小さい個体を優先的に選択する
            idx = 0
            for i in range(POPULATION):
                #適合度が同じのが現れるまでnextIndexにappendする
                if precision_list[sorted_indices[POPULATION]] == precision_list[sorted_indices[i]]:
                    break
                idx = idx + 1
                nextIndividuals.append(individuals[sorted_indices[i]])
            #指定の適合度(切れ目の個体の適合度)の個体のインデックスを保存するリスト
            precision_equal_idx = []
            for i in range(len(precision_list)):
                if precision_list[i] == precision_list[sorted_indices[idx]]:
                    precision_equal_idx.append(i)
            #リスクの小さい順にnextIndexにappendしていくためのリストを作る
            risk_min = []
            for idx in precision_equal_idx:
                risk_min.append(p[idx] + (idx,))
            #risk_minリストをリスクの小さい順に並び替える
            sorted_risk_min = sorted(risk_min, key=lambda x: x[0])
            idx = len(nextIndividuals)
            for row in sorted_risk_min:
                if idx == POPULATION:
                    break
                idx = idx + 1
                nextIndividuals.append(individuals[row[2]])
    #適合度0の個体が次世代に残す所定個体数以下である場合の処理終了
    else:
        #適合度0の個体が次世代に残す個体より多い場合
        #個体度0の個体のindividualsのインデックスをリストに格納
        next_idx = []
        for i in range(len(precision_list)):
            if precision_list[i] == 0:
                next_idx.append(i)

        #next_idxの要素数がpopulationと等しくなるまで繰り返す
        while True:
            if len(next_idx) == POPULATION:
                break
            #next_idxのすべての組み合わせに対して距離を保存するリスト
            #リストの形式は[(距離, 個体のインデックス1, 個体のインデックス2), ...]
            next_idx_distance = []
            #Step 1. next_idxのすべての組み合わせに対して距離を計算, リストに保存
            for i in next_idx:
                for j in next_idx:
                    if (i < j):
                        next_idx_distance.append((calculate_distance(p, i, j), i, j))
            #Step2. もっとも距離の短い2点を見つけ出しmin_distanceにタプル形式で代入する
            min_distance = min(next_idx_distance, key=lambda x: x[0])
            #Step3. もっとも近接している2個体それぞれの個体に対して, 2番目に近接している個体との距離を計算
            min_dis1 = []
            min_dis2 = []
            for i in range(len(next_idx)):
                if (i != min_distance[1]) and (i != min_distance[2]):
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
    individuals = copy.deepcopy(nextIndividuals)
    terminal = terminal + 1
    print(terminal)
#一世代での操作終了

print("最適化された銘柄の組み合わせ = ")
print(individuals)
print("最適化された個体の個体数")
print(len(individuals))

#リターンとリスクを保存するリスト, 行は個体に対応
returns = []
risks = []

#リターンを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
for individual in individuals:
    ret = 0
    #一つの個体のリターンを計算
    for i in individual:
        ret = ret + (r(i, stock_dict, T) * (1/LOCUS))
    returns.append(ret)
#リターンの計算終了

#リスクを計算 投資比率は1/locusで固定(個体内で均等な投資比率)
for individual in individuals:
    ris = 0
    for i in individual:
        for j in individual:
            ris = ris + (sigma(i, j, stock_dict, T, r(i, stock_dict, T), r(j, stock_dict, T)) * (1 / LOCUS) * (1 / LOCUS))
    risks.append(ris)
#リスクの計算終了

#[(risk, return), (risk, return), ...]に形を整える
p = [(x, y) for x, y in zip(risks, returns)]
#Evaluation終了
print("最適化された銘柄のリターンリスク = ")
print(p)
print("最適化されたリスクとリターンの個体数")
print(len(p))
