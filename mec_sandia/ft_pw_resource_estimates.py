import numpy
from math import factorial
from sympy import factorint

pv = numpy.array([[0.25, 0.25, 0.234375, 0.234375, 0.23046875, 0.23046875, 0.2294921875, 0.2294921875, 0.229248046875, 0.229248046875, 
    0.22918701171875, 0.22918701171875, 
    0.2291717529296875, 0.2291717529296875, 
    0.229167938232421875, 0.229167938232421875, 
    0.22916698455810546875, 0.22916698455810546875, 
    0.2291667461395263671875, 0.2291667461395263671875, 
    0.229166686534881591796875, 0.229166686534881591796875, 
    0.22916667163372039794921875, 
    0.22916667163372039794921875, 
    0.2291666679084300994873046875, 
    0.2291666679084300994873046875, 
    0.229166666977107524871826171875, 
    0.229166666977107524871826171875, 
    0.22916666674427688121795654296875, 
    0.22916666674427688121795654296875, 
    0.2291666666860692203044891357421875, 
    0.2291666666860692203044891357421875, 
    0.229166666671517305076122283935546875, 
    0.229166666671517305076122283935546875, 
    0.22916666666787932626903057098388671875, 
    0.22916666666787932626903057098388671875, 
    0.22916666666696983156725764274597167969, 
    0.22916666666696983156725764274597167969, 
    0.22916666666674245789181441068649291992, 
    0.22916666666674245789181441068649291992, 
    0.22916666666668561447295360267162322998, 
    0.22916666666668561447295360267162322998, 
    0.2291666666666714036182384006679058075, 
    0.2291666666666714036182384006679058075, 
    0.22916666666666785090455960016697645187, 
    0.22916666666666785090455960016697645187, 
    0.22916666666666696272613990004174411297, 
    0.22916666666666696272613990004174411297, 
    0.22916666666666674068153497501043602824, 
    0.22916666666666674068153497501043602824], [0.32421875, 
    0.28645833333333333333333333333333333333, 
    0.26041666666666666666666666666666666667, 0.24609375, 
    0.240966796875, 0.2386474609375, 0.23626708984375, 
    0.2354736328125, 0.23506673177083333333333333333333333333,
     0.23488362630208333333333333333333333333, 
    0.23477808634440104166666666666666666667, 
    0.2347011566162109375, 
    0.23466523488362630208333333333333333333, 
    0.23466046651204427083333333333333333333, 
    0.23465283711751302083333333333333333333, 
    0.23464949925740559895833333333333333333, 
    0.23464777072270711263020833333333333333, 
    0.23464680711428324381510416666666666667, 
    0.23464628557364145914713541666666666667, 
    0.2346460521221160888671875, 
    0.23464595278104146321614583333333333333, 
    0.23464592049519220987955729166666666667, 
    0.23464588976154724756876627604166666667, 
    0.23464587160075704256693522135416666667, 
    0.23464586252036194006601969401041666667, 
    0.23464586135620872179667154947916666667, 
    0.2346458598040044307708740234375, 
    0.234645858989097177982330322265625, 
    0.23464585870290951182444890340169270833, 
    0.23464585846765354896585146586100260417, 
    0.23464585831607109867036342620849609375, 
    0.2346458582687773741781711578369140625, 
    0.2346458582396735437214374542236328125, 
    0.23464585823057859670370817184448242188, 
    0.23464585822489425481762737035751342773, 
    0.23464585822076363304707532127698262532, 
    0.23464585821854673971150380869706471761, 
    0.23464585821811094016690428058306376139, 
    0.23464585821767514062230475246906280518, 
    0.23464585821740987133428764839967091878, 
    0.23464585821729263178288723186900218328, 
    0.23464585821724467014822342510645588239, 
    0.2346458582172135839035339207233240207, 
    0.23464585821720381394091721934576829274, 
    0.23464585821719966910829195209468404452, 
    0.23464585821719833684066240190683553616, 
    0.23464585821719680103214500377362128347, 
    0.23464585821719557053495937755845564728, 
    0.23464585821719499229380071862275750997, 
    0.23464585821719486739371044829264671231], [0.37806919642857142857142857142857142857, 0.30810546875, 
    0.27542550223214285714285714285714285714, 
    0.25503976004464285714285714285714285714, 
    0.24664306640625, 
    0.24206107003348214285714285714285714286, 
    0.2394866943359375, 0.23858642578125, 
    0.23798479352678571428571428571428571429, 
    0.23765918186732700892857142857142857143, 
    0.23747457776750837053571428571428571429, 
    0.23738268443516322544642857142857142857, 
    0.23734293665204729352678571428571428571, 
    0.23732791628156389508928571428571428571, 
    0.23731745992388044084821428571428571429, 
    0.23731321947915213448660714285714285714, 
    0.23731089915548052106584821428571428571, 
    0.237309582531452178955078125, 
    0.2373088784515857696533203125, 
    0.23730856765593801225934709821428571429, 
    0.23730839908655200685773577008928571429, 
    0.23730831579970461981637137276785714286, 
    0.23730827312517379011426653180803571429, 
    0.23730824832871024097715105329241071429, 
    0.23730823715283934559140886579241071429, 
    0.23730823386826419404574802943638392857, 
    0.23730823188296718788998467581612723214, 
    0.237308230833150446414947509765625, 
    0.23730823035761464520224503108433314732, 
    0.2373082300517645697774631636483328683, 
    0.23730822987103497975372842379978724888, 
    0.23730822979873015096278062888554164341, 
    0.23730822976777484914886632135936192104, 
    0.23730822975021835320928533162389482771, 
    0.23730822974056309249135665595531463623, 
    0.23730822973523199185105373284646442958, 
    0.23730822973227613407029171607324055263, 
    0.23730822973126208808025694452226161957, 
    0.23730822973060026827494896549199308668, 
    0.23730822973025057974284988761480365481, 
    0.23730822973009096139256663653733474868, 
    0.237308229730022888860828191224884774, 
    0.23730822972998901119824820073387984719, 
    0.23730822972997500652780900054494850338, 
    0.23730822972996704464268954585090146533, 
    0.23730822972996298836356743322539841756, 
    0.23730822972996051415225541144796547347, 
    0.23730822972995905301945336012587566594, 
    0.23730822972995844487496621054235999639, 
    0.23730822972995820151804429493885046603], [0.41328125, 
    0.31993815104166666666666666666666666667, 
    0.28391927083333333333333333333333333333, 
    0.26015828450520833333333333333333333333, 
    0.2495574951171875, 
    0.24412740071614583333333333333333333333, 
    0.24135920206705729166666666666666666667, 
    0.24005521138509114583333333333333333333, 
    0.2393444061279296875, 0.238973522186279296875, 
    0.23877418835957845052083333333333333333, 
    0.23867202599843343098958333333333333333, 
    0.23862770001093546549479166666666666667, 
    0.238606727123260498046875, 
    0.23859505454699198404947916666666666667, 
    0.23858956942955652872721354166666666667, 
    0.23858671387036641438802083333333333333, 
    0.23858523083229859670003255208333333333, 
    0.23858444156746069590250651041666666667, 
    0.23858408071100711822509765625, 
    0.23858388888960083325703938802083333333, 
    0.2385837950743734836578369140625, 
    0.2385837471694685518741607666015625, 
    0.23858372099348343908786773681640625, 
    0.23858370918217891206343968709309895833, 
    0.238583704541088081896305084228515625, 
    0.2385837020658073015511035919189453125, 
    0.23858370073321566451340913772583007812, 
    0.2385837000779550483760734399159749349, 
    0.23858369972413129289634525775909423828, 
    0.23858369954061042032359788815180460612, 
    0.23858369945652763514469067255655924479, 
    0.23858369941370180337495791415373484294, 
    0.23858369939045095028025874247153600057, 
    0.23858369937856688617709248016277949015, 
    0.23858369937209005229306058026850223541, 
    0.23858369936887697804195340722799301148, 
    0.23858369936751024908971885452046990395, 
    0.23858369936677158070400158370224138101, 
    0.23858369936639685822873010086671759685, 
    0.23858369936623025076016801904188469052, 
    0.23858369936615340112240346570615656674, 
    0.23858369936611182327013125359371770173, 
    0.23858369936609212236255928019090788439, 
    0.2385836993660816511090653572561374555, 
    0.23858369936607637153599033770963918262, 
    0.23858369936607351479337009910371610507, 
    0.2385836993660720355368380178144131302, 
    0.23858369936607134922240680552363301103, 
    0.23858369936607105058976041611629170802], [0.43584023752520161290322580645161290323, 0.3270263671875, 
    0.28919244581653225806451612903225806452, 
    0.26328720584992439516129032258064516129, 
    0.2513556941863029233870967741935483871, 
    0.2452950016144783266129032258064516129, 
    0.24224241318241242439516129032258064516, 
    0.24075715772567256804435483870967741935, 
    0.2399871810790031186995967741935483871, 
    0.23960470384167086693548387096774193548, 
    0.23940444184887793756300403225806451613, 
    0.23930164306394515498991935483870967742, 
    0.23925359547138214111328125, 
    0.23923036528210486135175151209677419355, 
    0.23921783652997785998928931451612903226, 
    0.23921181898443929610713835685483870968, 
    0.23920875703615526999196698588709677419, 
    0.23920713808207261946893507434475806452, 
    0.23920633270585488888525193737399193548, 
    0.23920595399930231994198214623235887097, 
    0.23920575385883210166808097593245967742, 
    0.23920565561274997889995574951171875, 
    0.23920560502278197917246049450289818548, 
    0.23920557915440557764903191597230972782, 
    0.23920556695262465866342667610414566532, 
    0.23920556162134274810312255736320249496, 
    0.2392055587959534492373706833008796938, 
    0.2392055573262393497110855194830125378, 
    0.23920555658593266637199708530979771768, 
    0.23920555620759753970384237266355945218, 
    0.2392055560189562364088957228006855134, 
    0.23920555592608417511303653760302451349, 
    0.23920555587772096919985826037103129971, 
    0.23920555585309401639308142025143869462, 
    0.23920555584040624441308130870663350628, 
    0.23920555583388936418765708015510632146, 
    0.23920555583063559417437758266685470458, 
    0.23920555582916654706819357553256615516, 
    0.23920555582840480514417527780507601077, 
    0.23920555582802569263218255136669012568, 
    0.23920555582784371275298810594963029988, 
    0.23920555582775678587152455175753622767, 
    0.23920555582771114496111867112866152198, 
    0.23920555582768828152954058403397099145, 
    0.23920555582767671596024978124396218317, 
    0.23920555582767106148918278032697931086, 
    0.23920555582766809553173002352962377641, 
    0.23920555582766656736625503702540637248, 
    0.23920555582766583150075343929926468205, 
    0.23920555582766548141536937009273940696], [0.44976080031622023809523809523809523809, 
    0.33109695192367311507936507936507936508, 
    0.29233333042689732142857142857142857143, 
    0.26504083663698226686507936507936507936, 
    0.25229472205752418154761904761904761905, 
    0.24590510413760230654761904761904761905, 
    0.24269946416219075520833333333333333333, 
    0.24112041223616827101934523809523809524, 
    0.24031784704753330775669642857142857143, 
    0.23991919793779887850322420634920634921, 
    0.23971589855731479705326140873015873016, 
    0.23961320188310411241319444444444444445, 
    0.23956280284457736545138888888888888889, 
    0.23953812566423226916600787450396825397, 
    0.23952518870669697958325582837301587302, 
    0.23951880143038810245574466765873015873, 
    0.23951558792401873876178075396825396825, 
    0.23951396560074672812507266090029761905, 
    0.23951316185458193695734417627728174603, 
    0.23951276896857760018772549099392361111, 
    0.23951256549076384140385521782769097222, 
    0.23951246457519508632166045052664620536, 
    0.23951241379894319892166152833000062004, 
    0.23951238831377561418487439079890175471, 
    0.23951237572189365025787126450311569941, 
    0.23951236987020473905085098175775437128, 
    0.23951236682783668844162353447505405971, 
    0.2395123652648916980806028559094383603, 
    0.23951236448039477210610158859737335689, 
    0.23951236408481104407410529102124865093, 
    0.23951236388903261004521655628368968055, 
    0.23951236379176348254842441602950058286, 
    0.23951236374127631959723542437016490906, 
    0.23951236371594470977159953835080303843, 
    0.2395123637031593955180863896015262793, 
    0.2395123636966897206340786757960265118, 
    0.23951236369348909215930342768496345906, 
    0.23951236369198337773490771319807714058, 
    0.23951236369120744021960903742255473007, 
    0.23951236369081708492302144775980166973, 
    0.23951236369062853799362928821354880855, 
    0.23951236369053620708880278323069057752, 
    0.23951236369048844530529752630411080428, 
    0.23951236369046493718634517996803431685, 
    0.23951236369045291279912733374755719394, 
    0.23951236369044699216725235495283911968, 
    0.23951236369044389681135284874188972112, 
    0.23951236369044233791793395581414137673, 
    0.23951236369044158181771129020479601682, 
    0.23951236369044121167496175426960190537], [0.45820740827425258366141732283464566929, 
    0.33344132130540262057086614173228346457, 
    0.29412954435573788139763779527559055118, 
    0.26601755712914654589074803149606299213, 
    0.25282741907074695497047244094488188976, 
    0.24624320090286375030757874015748031496, 
    0.24294791963156752699003444881889763779, 
    0.24131229120915330301119586614173228347, 
    0.24048949355684866116741510826771653543, 
    0.24007915617443445160632997047244094488, 
    0.23987183928137689124880813238188976378, 
    0.23976801604208514446348655880905511811, 
    0.2397164655556007633059043583907480315, 
    0.23969080965364659865071454386072834646, 
    0.23967772470163841416516641932209645669, 
    0.23967125261382119158121544545091043307, 
    0.2396679934229716423928268312469242126, 
    0.23966635355870716097786670594703494095, 
    0.23966553619859729138180965513694943405, 
    0.23966513159388069459420489513967919537, 
    0.23966492777471439766250257416972963829, 
    0.23966482598930950217887641876701294907, 
    0.23966477449737895072958483470706489143, 
    0.23966474861681203983034791908864899883, 
    0.23966473579451761849675562203399778351, 
    0.23966472959937629467298049391723993256, 
    0.23966472645037308075546844327074336255, 
    0.23966472485114883865902611003147335503, 
    0.2396647240508488898632904023342714535, 
    0.23966472365203260977526894755896151535, 
    0.23966472345263977157678817193515188112, 
    0.23966472335303797495013473447280253951, 
    0.23966472330213467237968856581545427559, 
    0.23966472327672879725073435683881379957, 
    0.23966472326398365547587705289580177252, 
    0.23966472325757653304921589329665504748, 
    0.23966472325438904646657571300768429106, 
    0.23966472325283649004544298174629706567, 
    0.23966472325204956494905394850957952623, 
    0.23966472325165419005262690703373692937, 
    0.23966472325146058726353351310206376434, 
    0.2396647232513653347175253040268574933, 
    0.23966472325131671384696290617265136877, 
    0.23966472325129249434124217636054694824, 
    0.23966472325128024848942217209694522747, 
    0.23966472325127416615023633083589381715, 
    0.23966472325127106811893482707977100057, 
    0.23966472325126951200368067604220073456, 
    0.23966472325126874677058595140559590373, 
    0.23966472325126836810049181180962862399]])

eps_mt = numpy.array([
    [
        10.666666666666666,
        4.666666666666667,
        2.4791666666666665,
        1.1770833333333333,
        0.6080729166666666,
        0.3001302083333333,
        0.15128580729166666,
        0.07539876302083333,
        0.037775675455729164,
        0.018872578938802082,
        0.009441057840983072,
        0.00471957524617513,
        0.002360085646311442,
        0.0011799832185109456,
        0.0005900102357069651,
        0.0002950013925631841,
        0.0001475018604348103,
        7.37506973867615e-05,
        3.687542145295689e-05,
        1.8437696174563218e-05,
        9.218852634755118e-06,
        4.609425407882857e-06,
        2.304712988158523e-06,
        1.1523564372358426e-06,
        5.761782363814897e-07,
        2.8808911463803116e-07,
        1.440445584292386e-07,
        7.20222789925747e-08,
        3.601113956567629e-08,
        1.8005569768960356e-08,
        9.002784888816988e-09,
        4.5013924435411325e-09,
        2.250696222041617e-09,
        1.1253481109665983e-09,
        5.626740555002398e-10,
        2.8133702774673177e-10,
        1.4066851387442465e-10,
        7.033425693700057e-11,
        3.516712846856646e-11,
        1.7583564234269994e-11,
        8.791782117139133e-12,
        4.39589105856874e-12,
        2.1979455292846285e-12,
        1.0989727646422624e-12,
        5.494863823211473e-13,
        2.7474319116057044e-13,
        1.3737159558028625e-13,
        6.868579779014291e-14,
        3.434289889507152e-14,
        1.7171449447535747e-14,
    ],
    [
        29.489504777701374,
        11.409961635658231,
        5.43295114614774,
        2.7976797289684288,
        1.4166793693262165,
        0.7010491711167368,
        0.3525316614071463,
        0.20268358671165948,
        0.0899999726166074,
        0.041868251093886207,
        0.024427325716614302,
        0.010774904779602763,
        0.00541318356620057,
        0.0032421455529227003,
        0.0013163804508143806,
        0.0007301602969405504,
        0.0003972481008009066,
        0.00018090699850510864,
        8.798842455367021e-05,
        4.819481105806367e-05,
        2.1221394821242915e-05,
        1.0198320553666686e-05,
        5.823037886843983e-06,
        2.905463250935223e-06,
        1.2906206897618548e-06,
        7.110268028483473e-07,
        3.1531598880351377e-07,
        1.6105434159801783e-07,
        9.856138539894317e-08,
        4.4626347375232446e-08,
        2.0642678764139505e-08,
        1.1811051576312408e-08,
        5.318269241572882e-09,
        2.726696300973302e-09,
        1.4216183939855642e-09,
        6.925275470374044e-10,
        3.2874274197547057e-10,
        1.8508419265127485e-10,
        8.489765202228366e-11,
        4.3176151561706244e-11,
        2.1883600561987743e-11,
        9.52819347154717e-12,
        5.053129340397738e-12,
        2.8553701560463323e-12,
        1.191358796292543e-12,
        6.036970025389076e-13,
        3.64260084891734e-13,
        1.7470269158225776e-13,
        8.133403855388398e-14,
        4.60981636709362e-14,
    ],
    [
        64.00328442638231,
        23.46944177816761,
        12.000371280913322,
        6.365104202612716,
        3.0744840314539488,
        1.5948403337115178,
        0.827662596766085,
        0.4639644700725331,
        0.21430999972805043,
        0.10118550725872207,
        0.05384662162634353,
        0.02520484849343578,
        0.012983589464160073,
        0.007071190770344802,
        0.0031304760080424457,
        0.0015861077785860355,
        0.0009138826369738761,
        0.0004228076074965519,
        0.00021001377137830593,
        0.00011465038229397025,
        5.0561725135682734e-05,
        2.4979310797198798e-05,
        1.3409597270461596e-05,
        6.815048002688419e-06,
        3.0750031392900574e-06,
        1.5727417317150848e-06,
        7.871073570075866e-07,
        3.896020439270433e-07,
        2.1111944784629813e-07,
        9.466351082268881e-08,
        4.5279327479822565e-08,
        2.5148496753171285e-08,
        1.299844826525357e-08,
        6.7107792305439e-09,
        3.11985788499786e-09,
        1.5874738992164303e-09,
        7.473467849955591e-10,
        4.2681935691305224e-10,
        2.0365897854504757e-10,
        1.013448385343065e-10,
        5.064423728518864e-11,
        2.3099704103364955e-11,
        1.2532144560784348e-11,
        6.61381039557453e-12,
        2.772397403049754e-12,
        1.398046270987868e-12,
        8.119414969427802e-13,
        4.029138025481759e-13,
        1.8380689006015555e-13,
        9.256648408639471e-14,
    ],
    [
        130.07307010943387,
        47.29464374043093,
        25.93563481256529,
        13.611857838016858,
        6.828468404288108,
        3.4658440125185925,
        1.7698467140614644,
        0.923511217939453,
        0.4468957802446324,
        0.2283260267202893,
        0.11648466177898728,
        0.05445760550094023,
        0.028403226161692472,
        0.0153330814609383,
        0.007410897184809677,
        0.003587215506244928,
        0.001871090105888392,
        0.0008893670613791814,
        0.0004595488607374706,
        0.00023872082063362444,
        0.00011033317893112022,
        5.6581568035169004e-05,
        2.9024022536471023e-05,
        1.4735449080669863e-05,
        6.787976093166994e-06,
        3.476052016860671e-06,
        1.7593024206351639e-06,
        9.032293467231383e-07,
        4.53431042052669e-07,
        2.1648116161645572e-07,
        1.0909715622460782e-07,
        5.623314950709884e-08,
        2.9365937833882423e-08,
        1.460102108016436e-08,
        7.104807770229208e-09,
        3.594533794945757e-09,
        1.6697977678551547e-09,
        8.948240967677716e-10,
        4.380513937363725e-10,
        2.2122631111584247e-10,
        1.0885868428689607e-10,
        5.204877408055106e-11,
        2.784232619631671e-11,
        1.4829843199757435e-11,
        6.468027787458967e-12,
        3.3073432933968113e-12,
        1.7990827250053349e-12,
        8.986740817202572e-13,
        4.1465485114450846e-13,
        2.0652802217868613e-13,
    ],
    [
        259.20267182347027,
        94.27696897143444,
        54.549887252505954,
        28.46565925149864,
        14.56716907448622,
        7.39613557361652,
        3.7517952520741544,
        1.9128673040145576,
        0.9322378006058091,
        0.47689402487513627,
        0.23846844554564237,
        0.11767064418376781,
        0.05964774040082835,
        0.031138920020541628,
        0.01560624678238143,
        0.007545584263798574,
        0.0038952910085483738,
        0.0019123105564021815,
        0.000973142851881326,
        0.0004911008340618087,
        0.0002334046662677894,
        0.00011882636324285214,
        6.045969965343745e-05,
        3.056034950415763e-05,
        1.445352520961731e-05,
        7.276060302695516e-06,
        3.645665960981373e-06,
        1.8975250364918994e-06,
        9.550074449705673e-07,
        4.5938632456974733e-07,
        2.3315935897197518e-07,
        1.1983184347557885e-07,
        6.090358977404524e-08,
        3.017931311862247e-08,
        1.490645028915272e-08,
        7.483987136617716e-09,
        3.54304260462219e-09,
        1.838899183865534e-09,
        9.281094346797382e-10,
        4.658087248738759e-10,
        2.2429755610183488e-10,
        1.1182077661999636e-10,
        5.870910584208697e-11,
        3.0319042321838956e-11,
        1.3931581359467513e-11,
        7.173503432740275e-12,
        3.69133975687558e-12,
        1.8461331744778614e-12,
        8.939864728033136e-13,
        4.397672773196028e-13,
    ],
    [
        514.5444012787318,
        187.993745316196,
        112.5688694789333,
        58.66195319154966,
        30.26955320806803,
        15.418445206584266,
        7.802980400007618,
        3.948778594680197,
        1.9552639303187647,
        0.9899569869287389,
        0.4941607339863597,
        0.24529481622372054,
        0.12359695940391202,
        0.06323599801113533,
        0.03164294934212927,
        0.015584003933747783,
        0.00791333162687218,
        0.003916841802870193,
        0.0019781555879535657,
        0.0009984659380669737,
        0.00048613925458710245,
        0.0002445196247659082,
        0.00012309642663603748,
        6.250912009651843e-05,
        3.0065321242149537e-05,
        1.5203520130171626e-05,
        7.612280386015644e-06,
        3.906087984293318e-06,
        1.9531470087652805e-06,
        9.55914363124602e-07,
        4.805373083623706e-07,
        2.4493551454525607e-07,
        1.2398460541808522e-07,
        6.067560725450882e-08,
        3.038453641442461e-08,
        1.5268852602405413e-08,
        7.35539561202542e-09,
        3.7539992282141205e-09,
        1.888318348828168e-09,
        9.515357721797175e-10,
        4.641454499084328e-10,
        2.3257852319174984e-10,
        1.18877550595156e-10,
        6.069912319996679e-11,
        2.8872199528195772e-11,
        1.4761319348005826e-11,
        7.430974628329412e-12,
        3.730393757734435e-12,
        1.83278554134611e-12,
        9.07907212461658e-13,
    ],
    [
        1022.1614805187997,
        374.98305637724195,
        229.42480434776226,
        119.55297239220033,
        61.94609610795401,
        31.60031322723858,
        15.98410147613242,
        8.0614202814249,
        4.012494385102454,
        2.0208279699457394,
        1.0112516873352073,
        0.5039790317053261,
        0.2521864494159917,
        0.1276511080647678,
        0.06373854373488438,
        0.031771248470686124,
        0.016091039436991257,
        0.007968980681165694,
        0.004009963209490315,
        0.002007596453729596,
        0.0009920791651935237,
        0.0004968935773968603,
        0.000250036686547759,
        0.00012600908341409663,
        6.180076165614051e-05,
        3.100087686004089e-05,
        1.5528737824860464e-05,
        7.85372503213337e-06,
        3.926970237607599e-06,
        1.9423212692126662e-06,
        9.784420067252215e-07,
        4.940652521203564e-07,
        2.4794866022544026e-07,
        1.220823437315778e-07,
        6.118543526641172e-08,
        3.0709055412288623e-08,
        1.5111982692340512e-08,
        7.650144542335827e-09,
        3.840897597976265e-09,
        1.927312621325955e-09,
        9.455921515700106e-10,
        4.744204171226399e-10,
        2.395270214663177e-10,
        1.2117033607700343e-10,
        5.923885800939487e-11,
        2.9836051059331784e-11,
        1.4972070619171786e-11,
        7.504219697571095e-12,
        3.729611890771701e-12,
        1.8512957785978144e-12,
    ],
])


def M1(k, K):
    return numpy.ceil(numpy.log2(factorial(K) * numpy.sum([1/factorial(k1) for  k1 in range(k,  K+1)]
                                                )
                          )
                   )

def g1(x, n):
    asin_val = 0.5 / numpy.sqrt(x)
    floored_val = numpy.floor(2**n * asin_val / (2 * numpy.pi))
    return floored_val * 2 * numpy.pi / 2**n

def h1(x, n):
    return x * ((1 + (2 - 4 * x) * numpy.sin(g1(x, n))**2)**2 + 4*numpy.sin(g1(x, n))**2 * numpy.cos(g1(x, n))**2)

def g2(x, n):
    asin_val = numpy.arcsin(0.5 / numpy.sqrt(x)) 
    return numpy.ceil(2**n * asin_val / (2 * numpy.pi)) * 2 * numpy.pi / 2**n

def h2(x, n):
    return x * ((1 + (2 - 4 * x) * numpy.sin(g2(x, n))**2)**2 + 4 * numpy.sin(g2(x, n))**2 * numpy.cos(g2(x, n))**2)

def h(x, n):
    return numpy.max([h1(x, n), h2(x, n)])

def Eq(n, br):
    return h(n / 2**(numpy.ceil(numpy.log2(n))), br)
    
def Er(zeta):
    kt1 = 2**numpy.floor(numpy.log2(zeta)/2)
    kt2 = 2**numpy.ceil(numpy.log2(zeta)/2)
    return numpy.min([numpy.ceil(zeta / kt1) + kt1, 
                   numpy.ceil(zeta / kt2) + kt2]
                 )

# Probability of success for creating the superposition over 3 basis states
Peq0 = Eq(3, 8)

def pw_qubitization_costs(np, eta, Omega, eps, nMc, nbr, L):
    """
    :params:
       lam_zeta is the sum over nuclear weights
       np is the number of bits in each direction for the momenta
       eta is the number of electrons
       rs is the Wigner-Seitz radius
       eps is the total allowable error
       pv is the precomputed vector of probabilities of success for the nu preparation
       eps_mt is the precomputed discretisation errors for the nu preparation
       nMc is an adjustment for the number of bits for M (used in nu preparation
       ntc is an adjustment in the number of bits used for the time
       nbr is an adjustment in the number of bits used for the nuclear positions
       L is the number of nuclei
    """
    # Total nuclear charge assumed to be equal to number of electrons. 
    lam_zeta = eta  
    
    # (*This is the number of bits used in rotations in preparations of equal superposition states.
    br = 7 
    
    # The following uses the precomputed table to find the exact value of p based on np.
    
    # (*Probability of success for creating the superposition over i and j.*)
    Peq1 = Eq(eta, br)**2
    
    # (*Probability of success for creating the equal superposition for the selection between U and V.*)
    Peq3 = Peq0; 
    
    # This uses pvec from planedata.nb, which is precomputed values for
    #  \[Lambda]_\[Nu]. We start with a very large  guess for the number 
    # of bits to use for M (precision in \[Nu] \ preparation) then adjust it.*)
    p = pv[np-1, 49]
    
    # (*Now compute the lambda-values.*)
    # (*Here 64*(2^np-1))*p is \[Lambda]_\[Nu].*)
    tmp = (64*(2**np - 1)) * p * eta / (2 * numpy.pi * Omega**(1/3))
    
    # (*See Eq. (D31) or (25).*)
    lam_UV = tmp * (eta - 1 + 2 * lam_zeta)
    
    # (*See Eq. (25), possibly should be replaced with expression from Eq. (71).*)
    lam_T =  6 * eta * numpy.pi**2 / Omega**(2/3) * (2**(np - 1) - 1)**2
    
    # (*Adjust value of nM based on \[Lambda]UV we just calculated.*) 
    nM = nMc + int(numpy.rint(numpy.log2(20 * lam_UV / eps)))
    
    #  (*Recompute p and \[Lambda]V.*)
    p = pv[np-1, nM-1] 
    lam_V = tmp * (eta - 1)
    lam_U = tmp * 2 * lam_zeta
    
    # (*See Eq. (117).*)
    pamp = numpy.sin(3*numpy.arcsin(numpy.sqrt(p)))  
    
    # (*We estimate the error due to the finite M using the precomputed table.*)
    epsM = eps_mt[np-1, nM-1] * eta * (eta - 1) / (2 * numpy.pi * Omega**(1/3))
    
    # (*First we estimate the error due to the finite precision of the \
    # nuclear positions. The following formula is from the formula for the \
    # error due to the nuclear positions in Theorem 4, where we have used \
    # (64*(2^np-1))*p for the sum over 1/|\[Nu]|.  First we estimate the \
    # number of bits to obtain an error that is some small fraction of the \
    # total error, then use that to compute the actual bound in the error \
    # for that number of bits.*)
    nrf = (64*(2**np - 1)) * p * eta * lam_zeta / Omega**(1/3)
    nR = nbr + numpy.rint(numpy.log2(nrf/eps));
    
    #  (*See Eq. (133).*)
    epsR =  nrf/2**nR  
    # (*Set the allowable error in the phase measurement such that the sum of the squares in the errors is \[Epsilon]^2, as per Eq. (131).*)
    
    if eps > epsM + epsR:
        eps_ph = numpy.sqrt(eps**2 - (epsM + epsR)**2)
    else:
        eps_ph = 10**(-100)
    #print(eps_ph, epsM, epsR)
    # (*The number of iterations of the phase measurement.*)
    
    # # (*See Eq. (127).*) 
    lam_1 = max(lam_T + lam_U + lam_V, (lam_U + lam_V / (1 - 1 / eta)) / p) / (Peq0 * Peq1* Peq3) 
    lam_2 = max(lam_T + lam_U + lam_V, (lam_U + lam_V / (1 - 1 / eta)) /pamp) / (Peq0 * Peq1 * Peq3)
    # (*See Eq. (126).*)
    # (*The P_eq is from Eq. (130), with P_s(\[Eta]+2lam_zeta) replaced with P_s(3,8). This is because we are taking \[Eta]=lam_zeta.*)
    #  (*Steps for phase estimation without amplitude amplification.*)
    m1 = numpy.ceil(numpy.pi * lam_1 / (2 * eps_ph)) 
    m2 = numpy.ceil(numpy.pi * lam_2 / (2 * eps_ph)) 
    # (*Steps for phase estimation with amplitude amplification.*)

    # (*The number of bits used for the equal state preparation for choosing between U and V.*)
    n_eta_zeta = numpy.ceil(numpy.log2(eta + lam_zeta))
    n_eta = numpy.ceil(numpy.log2(eta));
    # (*Set the costs of the parts of the block encoding according to the list in table II.*)

    # (*c1=2*(5*n\[Eta]\[Zeta]+2*br-9);
    # We instead compute the complexity according to the complexity of \
    # preparing an equal superposition for 3 basis states, plus the \
    # complexity of rotating a qubit for T.*)
    c1 = 2 * (n_eta_zeta + 13)
    # (*c2=14*n\[Eta]+8*br-36;*)
    factors = factorint(eta)
    bts = factors[min(list(sorted(factors.keys())))]
    # bts = FactorInteger[\[Eta]][[1, 2]];
    if eta % 2 > 0:
        bts = 0

    # (*This is cost of superposition over i and j. See Eq. (62), or table line 2.*)
    c2 = 14 * n_eta + 8 * br - 36 - 12 * bts
    # (*Table line 3.*)
    c3 = 2 * (2 * np + 9)
    # (*Table line 4.*)
    c4 = 12 * eta * np + 4 * eta - 8  
    # (*Table line 5.*)
    c5 = 5 * (np - 1) + 2  
    # (*Table line 6, modified?.*)
    c6 = 3 * np**2 + 13 * np - 10 + 2 * nM * (2 * np + 2)  

    # (*The QROM cost according to the number of nuclei, line 7 modified.*)
    c7 = L + Er(L)
    # (*Line 8.*)
    c8 = 24 * np
    #  (*See Eq. (97).*)

    # c9 = 3*(Piecewise[{{2*np*nR - np*(np + 1) - 1, nR > np}}, nR*(nR - 1)]) 
    c9 = 3 * (2*np*nR - np*(np + 1) - 1 if nR > np else nR*(nR - 1))


    # (*The number of qubits we are reflecting on according to equation (136).*)
    cr = n_eta_zeta + 2 * n_eta + 6*np + nM + 16

    # (*First the cost without the amplitude amplification.*)
    cq = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + cr) 
    # (*Next the cost with the amplitude amplification.*) 
    cqaa = (c1 + c2 + c3 + c4 + c5 + 3*c6 + c7 + c8 + c9 + cr) 

    # (*Qubits for qubitisation.*)
    q1 = 3 * eta * np # (*Qubits storing the momenta.*)

    # (*Qubits for phase estimation.*)
    # q2 = 2*numpy.ceil(numpy.log2(Piecewise[{{m1, cq < cqaa}}, m2]]] - 1 
    q2 = 2*numpy.ceil(numpy.log2(m1 if cq < cqaa else m2)) - 1

    # (*We are costing WITH nuclei, so the maximum precision of rotations is nR+1.*)
    q3 = nR + 1 
    # (*The |T> state.*)
    q4 = 1 
    # (*The rotated qubit for T vs U+V.*)
    q5 = 1 
    # (*The superposition state for selecting between U and V. This is changed from n\[Eta]\[Zeta]+3 to bL+4, with Log2[L] for outputting L.*)
    q6 = numpy.ceil(numpy.log2(L)) + 4 

    # (*The result of a Toffoli on the last two.*)
    q7 = 1 

    # (*Preparing the superposition over i and j.*)
    q8 = 2 * n_eta + 5 
    # (*For preparing the superposition over \[Nu].*)
    q9 = 3*(np + 1) + np + nM + (3*np + 2) + (2*np + 1) + (3*np^2 - np - 1 + 4*nM*(np + 1)) + 1 + 2

    # (*The nuclear positions.*)
    q10 = 3*nR 
    # (*Preparation of w.*)
    q11 = 4 
    # (*Preparation of w, r and s.*)
    q12 =2*np + 4 
    # (*Temporary qubits for updating momenta.*)
    q13 = 5*np + 1 
    # (*Overflow bits for arithmetic on momenta.*)
    q14 = 6
    # (*Arithmetic for phasing for nuclear positions.*)
    q15 = 2*(nR - 2) 
    qt = q1 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13 + q14

    final_cost_toffoli = cq if cq * m1 < cqaa * m2 else cqaa
    return final_cost_toffoli, qt