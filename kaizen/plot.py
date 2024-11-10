from array import array
import matplotlib.pyplot as plt

initial_combinations = ['1332', '4004', '8306', '5020', '4005', '4004', '4502', '1801', '5019', '5020', '7201', '4004', '4751', '4502', '4004', '8306', '5020', '2503', '1332', '4005', '7203', '6501', '8306', '4324', '4385', '2501', '1332', '5019', '4502', '6501', '5020', '1802', '4005', '1801', '4324', '5019', '2501', '4324', '5020', '1802', '5020', '4005', '4004', '1802', '1332', '5020', '6501', '4661', '1802', '4751', '7201', '4661', '4502', '7203', '1801', '6503', '1802', '4385', '4751', '1801', '4503', '2501', '1332', '1802', '4004', '4385', '4503', '4661', '5019', '4324', '4004', '8306', '4503', '4385', '4005', '4502', '4004', '1801', '2501', '1332', '4661', '4324', '4005', '4004', '4385', '4324', '8306', '7201', '1802', '4751', '5019', '7201', '1801', '4005', '4324', '6503', '6501', '8306', '7201', '2503', '2503', '7201', '4324', '4004', '7201', '6503', '1802', '4751', '8306', '2501', '1332', '6503', '4005', '5020', '4661', '5019', '4324', '7201', '2501', '1802', '4385', '2501', '6503', '4751', '2501', '4503', '5019', '4385', '1801', '4385', '2503', '4004', '4005', '4661', '4324', '4004', '1801', '7201', '4004', '2501', '4324', '7201', '7203', '1802', '2503', '7201', '4005', '1802', '4385', '4751', '6501', '4005', '4502', '6501', '6503', '4503', '4324', '1802', '6503', '4503', '4385', '2501', '1802', '5020', '4324', '4005', '6503', '8306', '4004', '4502', '1802', '2503', '4385', '4004', '1801', '4005', '7203', '4324', '4661', '1332', '2503', '5019', '4661', '5020', '4005', '5019', '7201', '4004', '4502', '1802', '1332', '2501', '4661', '1332', '1801', '1802', '4661', '1332', '5020', '7201']
initial_risk_array = array('f', [0.00013044947991147637, 0.0001155243007815443, 0.0001917141053127125, 0.00011386118421796709, 8.248084486695006e-05, 0.00013806932838633657, 0.00012852204963564873, 0.00010364072659285739, 0.00012304892879910767, 0.00010299700807081535, 0.0001461830543121323, 9.789725299924612e-05, 0.00013878302706871182, 0.00011564664600882679, 0.00016582351236138493, 0.00010481786739546806, 0.00013837337610311806, 0.0001380994071951136, 0.0001152348195319064, 0.00011170708603458479, 0.00011910233297385275, 0.00018817049567587674, 0.00015789535245858133, 0.00012099995365133509, 0.0001292150845983997, 0.00012272040476091206, 0.0001572171604493633, 0.00011461188842076808, 0.00013323788880370557, 0.0001301106676692143, 0.00015761397662572563, 0.00012964144116267562, 0.00012320517271291465, 0.00011910233297385275, 0.0001474435266572982, 0.0001633668871363625, 0.00012261682422831655, 0.00015232060104608536, 0.0001061764924088493, 0.00010889511759160087, 0.00012573236017487943, 0.00012311909813433886, 8.204177720472217e-05, 0.00015515378618147224, 9.937711001839489e-05, 0.00010121364903170615, 0.00018543995975051075, 9.16917560971342e-05, 0.00010948652197839692, 0.0001204018117277883])
initial_return_array = array('f', [0.0013519433559849858, 0.0003001034492626786, 0.0013615201460197568, 0.000500349560752511, 0.00043752833153121173, 0.0010982491075992584, 0.0014913197373971343, 0.0010254771914333105, 0.00013634956849273294, 0.0011400923831388354, 0.0006320287357084453, 0.0014994684606790543, 0.0007568557048216462, 0.0010635912185534835, 0.00025906405062414706, 0.0012308672303333879, 0.000695694237947464, 0.0009414096712134778, -0.0003157324972562492, 0.001651085796765983, 0.0003723606641869992, 0.0006259975489228964, 0.0007318827556446195, 0.0002562272420618683, 0.0012165338266640902, 0.0007046109531074762, 0.0008517603855580091, 0.0018715913174673915, 0.0007088767015375197, 0.0012225359678268433, 0.001014573615975678, 0.0010023360373452306, 0.0006775677902624011, 0.0003723606641869992, 0.0016324601601809263, 0.0009123069467023015, 0.0003940792812500149, 3.520659083733335e-05, 0.0008504194556735456, 0.0004993368638679385, 0.001297271461226046, 0.0003979590837843716, 0.0007199938409030437, 0.0003346326993778348, 0.0010488844709470868, 0.001051811850629747, 0.0008074513752944767, 0.0013638115487992764, 0.0011692570988088846, 0.0013783029280602932])
optimized_combinations = ['4502', '4661', '1332', '2503', '2501', '4502', '5020', '2503', '2501', '4661', '4502', '2503', '2501', '4502', '1332', '2503', '2501', '4502', '4004', '2503', '2501', '5020', '4661', '2503', '2501', '5020', '1332', '2503', '2501', '4502', '4661', '1332', '2501', '4661', '1332', '2503', '2501', '7203', '4661', '2503', '2501', '7203', '1332', '2503', '2501', '6501', '4661', '2503', '2501', '6501', '1332', '2503', '2501', '5020', '4661', '1332', '2501', '7203', '4661', '1332', '2501', '4661', '6503', '1332', '2501', '4661', '6501', '1332', '2501', '7203', '6503', '1332', '2501', '7203', '6501', '1332', '2501', '6501', '6503', '1332', '2501', '6501', '6503', '4004', '4502', '4661', '1332', '2503', '2501', '4502', '5020', '2503', '2501', '4661', '4502', '2503', '2501', '4502', '1332', '2503', '2501', '4502', '4004', '2503', '2501', '5020', '4661', '2503', '2501', '5020', '1332', '2503', '2501', '4502', '4661', '1332', '2501', '4661', '1332', '2503', '2501', '7203', '4661', '2503', '2501', '7203', '1332', '2503', '2501', '6501', '4661', '2503', '2501', '6501', '1332', '2503', '2501', '5020', '4661', '1332', '2501', '7203', '4661', '1332', '2501', '4661', '6503', '1332', '2501', '6501', '4661', '1332', '2501', '7203', '1332', '6503', '2501', '7203', '6501', '1332', '2501', '6501', '6503', '1332', '2501', '6501', '6503', '4004', '4502', '4661', '1332', '2503', '2501', '4502', '5020', '2503', '2501', '4502', '4661', '2503', '2501', '4502', '1332', '2503', '2501', '4502', '4004', '2503', '2501', '4661', '5020', '2503', '2501', '5020', '1332', '2503', '2501', '4502', '4661', '1332']
optimized_risk_array = array('f', [6.257827772060409e-05, 6.816950917709619e-05, 7.000016194069758e-05, 7.485720561817288e-05, 7.542167440988123e-05, 7.626982551300898e-05, 8.21063295006752e-05, 8.29903146950528e-05, 8.35144892334938e-05, 8.614465332357213e-05, 8.792551670921966e-05, 8.951849304139614e-05, 8.961190906120464e-05, 9.067958308150992e-05, 9.618245530873537e-05, 0.00010131709859706461, 0.0001013958899420686, 0.00010933864541584626, 0.00010988349094986916, 0.0001107095304178074, 0.00013070057320874184, 6.257827772060409e-05, 6.816950917709619e-05, 7.000016194069758e-05, 7.485720561817288e-05, 7.542167440988123e-05, 7.626982551300898e-05, 8.21063295006752e-05, 8.29903146950528e-05, 8.35144892334938e-05, 8.614465332357213e-05, 8.792551670921966e-05, 8.951849304139614e-05, 8.961190906120464e-05, 9.067958308150992e-05, 9.618245530873537e-05, 0.00010131709859706461, 0.0001013958899420686, 0.00010933864541584626, 0.00010988349094986916, 0.0001107095304178074, 0.00013070057320874184, 6.257827772060409e-05, 6.816950917709619e-05, 7.000016194069758e-05, 7.485720561817288e-05, 7.542167440988123e-05, 7.626982551300898e-05, 8.21063295006752e-05, 8.29903146950528e-05])
optimized_return_array = array('f', [0.0008297555614262819, 0.001076432061381638, 0.0011695604771375656, 0.0011775012826547027, 0.001185691449791193, 0.0014245305210351944, 0.0014324713265523314, 0.0014817634364590049, 0.0015255998587235808, 0.0015551757533103228, 0.0015631165588274598, 0.0016281397547572851, 0.0016360805602744222, 0.0017367334803566337, 0.001867378712631762, 0.001924766693264246, 0.0019403427140787244, 0.001962283393368125, 0.0019778592977672815, 0.0020352473948150873, 0.0020434376783668995, 0.0008297555614262819, 0.001076432061381638, 0.0011695604771375656, 0.0011775012826547027, 0.001185691449791193, 0.0014245305210351944, 0.0014324713265523314, 0.0014817634364590049, 0.0015255998587235808, 0.0015551757533103228, 0.0015631165588274598, 0.0016281397547572851, 0.0016360805602744222, 0.0017367334803566337, 0.001867378712631762, 0.001924766693264246, 0.0019403427140787244, 0.001962283393368125, 0.0019778592977672815, 0.0020352473948150873, 0.0020434376783668995, 0.0008297555614262819, 0.001076432061381638, 0.0011695604771375656, 0.0011775012826547027, 0.001185691449791193, 0.0014245305210351944, 0.0014324713265523314, 0.0014817634364590049])

# グラフのプロット
plt.scatter(initial_risk_array, initial_return_array, color='red', label='initial')
plt.scatter(optimized_risk_array, optimized_return_array, color='blue', label='optimized')

# ラベルとタイトルの設定
plt.xlabel("risk")
plt.ylabel("return")
plt.title("Initial population and Optimization population")
plt.legend()
plt.grid()

# グラフの表示
plt.show()
