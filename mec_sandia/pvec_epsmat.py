import numpy 
pvec = numpy.array([[0.25000000000000000000, 0.25000000000000000000, 
  0.23437500000000000000, 0.23437500000000000000, 
  0.23046875000000000000, 0.23046875000000000000, 
  0.22949218750000000000, 0.22949218750000000000, 
  0.22924804687500000000, 0.22924804687500000000, 
  0.22918701171875000000, 0.22918701171875000000, 
  0.22917175292968750000, 0.22917175292968750000, 
  0.22916793823242187500, 0.22916793823242187500, 
  0.22916698455810546875, 0.22916698455810546875, 
  0.22916674613952636719, 0.22916674613952636719, 
  0.22916668653488159180, 0.22916668653488159180, 
  0.22916667163372039795, 0.22916667163372039795, 
  0.22916666790843009949, 0.22916666790843009949, 
  0.22916666697710752487, 0.22916666697710752487, 
  0.22916666674427688122, 0.22916666674427688122, 
  0.22916666668606922030, 0.22916666668606922030, 
  0.22916666667151730508, 0.22916666667151730508, 
  0.22916666666787932627, 0.22916666666787932627, 
  0.22916666666696983157, 0.22916666666696983157, 
  0.22916666666674245789, 0.22916666666674245789, 
  0.22916666666668561447, 0.22916666666668561447, 
  0.22916666666667140362, 0.22916666666667140362, 
  0.22916666666666785090, 0.22916666666666785090, 
  0.22916666666666696273, 0.22916666666666696273, 
  0.22916666666666674068, 
  0.22916666666666674068], [0.32421875000000000000, 
  0.28645833333333333333, 0.26041666666666666667, 
  0.24609375000000000000, 0.24096679687500000000, 
  0.23864746093750000000, 0.23626708984375000000, 
  0.23547363281250000000, 0.23506673177083333333, 
  0.23488362630208333333, 0.23477808634440104167, 
  0.23470115661621093750, 0.23466523488362630208, 
  0.23466046651204427083, 0.23465283711751302083, 
  0.23464949925740559896, 0.23464777072270711263, 
  0.23464680711428324382, 0.23464628557364145915, 
  0.23464605212211608887, 0.23464595278104146322, 
  0.23464592049519220988, 0.23464588976154724757, 
  0.23464587160075704257, 0.23464586252036194007, 
  0.23464586135620872180, 0.23464585980400443077, 
  0.23464585898909717798, 0.23464585870290951182, 
  0.23464585846765354897, 0.23464585831607109867, 
  0.23464585826877737418, 0.23464585823967354372, 
  0.23464585823057859670, 0.23464585822489425482, 
  0.23464585822076363305, 0.23464585821854673971, 
  0.23464585821811094017, 0.23464585821767514062, 
  0.23464585821740987133, 0.23464585821729263178, 
  0.23464585821724467015, 0.23464585821721358390, 
  0.23464585821720381394, 0.23464585821719966911, 
  0.23464585821719833684, 0.23464585821719680103, 
  0.23464585821719557053, 0.23464585821719499229, 
  0.23464585821719486739], [0.37806919642857142857, 
  0.30810546875000000000, 0.27542550223214285714, 
  0.25503976004464285714, 0.24664306640625000000, 
  0.24206107003348214286, 0.23948669433593750000, 
  0.23858642578125000000, 0.23798479352678571429, 
  0.23765918186732700893, 0.23747457776750837054, 
  0.23738268443516322545, 0.23734293665204729353, 
  0.23732791628156389509, 0.23731745992388044085, 
  0.23731321947915213449, 0.23731089915548052107, 
  0.23730958253145217896, 0.23730887845158576965, 
  0.23730856765593801226, 0.23730839908655200686, 
  0.23730831579970461982, 0.23730827312517379011, 
  0.23730824832871024098, 0.23730823715283934559, 
  0.23730823386826419405, 0.23730823188296718789, 
  0.23730823083315044641, 0.23730823035761464520, 
  0.23730823005176456978, 0.23730822987103497975, 
  0.23730822979873015096, 0.23730822976777484915, 
  0.23730822975021835321, 0.23730822974056309249, 
  0.23730822973523199185, 0.23730822973227613407, 
  0.23730822973126208808, 0.23730822973060026827, 
  0.23730822973025057974, 0.23730822973009096139, 
  0.23730822973002288886, 0.23730822972998901120, 
  0.23730822972997500653, 0.23730822972996704464, 
  0.23730822972996298836, 0.23730822972996051415, 
  0.23730822972995905302, 0.23730822972995844487, 
  0.23730822972995820152], [0.41328125000000000000, 
  0.31993815104166666667, 0.28391927083333333333, 
  0.26015828450520833333, 0.24955749511718750000, 
  0.24412740071614583333, 0.24135920206705729167, 
  0.24005521138509114583, 0.23934440612792968750, 
  0.23897352218627929688, 0.23877418835957845052, 
  0.23867202599843343099, 0.23862770001093546549, 
  0.23860672712326049805, 0.23859505454699198405, 
  0.23858956942955652873, 0.23858671387036641439, 
  0.23858523083229859670, 0.23858444156746069590, 
  0.23858408071100711823, 0.23858388888960083326, 
  0.23858379507437348366, 0.23858374716946855187, 
  0.23858372099348343909, 0.23858370918217891206, 
  0.23858370454108808190, 0.23858370206580730155, 
  0.23858370073321566451, 0.23858370007795504838, 
  0.23858369972413129290, 0.23858369954061042032, 
  0.23858369945652763514, 0.23858369941370180337, 
  0.23858369939045095028, 0.23858369937856688618, 
  0.23858369937209005229, 0.23858369936887697804, 
  0.23858369936751024909, 0.23858369936677158070, 
  0.23858369936639685823, 0.23858369936623025076, 
  0.23858369936615340112, 0.23858369936611182327, 
  0.23858369936609212236, 0.23858369936608165111, 
  0.23858369936607637154, 0.23858369936607351479, 
  0.23858369936607203554, 0.23858369936607134922, 
  0.23858369936607105059], [0.43584023752520161290, 
  0.32702636718750000000, 0.28919244581653225806, 
  0.26328720584992439516, 0.25135569418630292339, 
  0.24529500161447832661, 0.24224241318241242440, 
  0.24075715772567256804, 0.23998718107900311870, 
  0.23960470384167086694, 0.23940444184887793756, 
  0.23930164306394515499, 0.23925359547138214111, 
  0.23923036528210486135, 0.23921783652997785999, 
  0.23921181898443929611, 0.23920875703615526999, 
  0.23920713808207261947, 0.23920633270585488889, 
  0.23920595399930231994, 0.23920575385883210167, 
  0.23920565561274997890, 0.23920560502278197917, 
  0.23920557915440557765, 0.23920556695262465866, 
  0.23920556162134274810, 0.23920555879595344924, 
  0.23920555732623934971, 0.23920555658593266637, 
  0.23920555620759753970, 0.23920555601895623641, 
  0.23920555592608417511, 0.23920555587772096920, 
  0.23920555585309401639, 0.23920555584040624441, 
  0.23920555583388936419, 0.23920555583063559417, 
  0.23920555582916654707, 0.23920555582840480514, 
  0.23920555582802569263, 0.23920555582784371275, 
  0.23920555582775678587, 0.23920555582771114496, 
  0.23920555582768828153, 0.23920555582767671596, 
  0.23920555582767106149, 0.23920555582766809553, 
  0.23920555582766656737, 0.23920555582766583150, 
  0.23920555582766548142], [0.44976080031622023810, 
  0.33109695192367311508, 0.29233333042689732143, 
  0.26504083663698226687, 0.25229472205752418155, 
  0.24590510413760230655, 0.24269946416219075521, 
  0.24112041223616827102, 0.24031784704753330776, 
  0.23991919793779887850, 0.23971589855731479705, 
  0.23961320188310411241, 0.23956280284457736545, 
  0.23953812566423226917, 0.23952518870669697958, 
  0.23951880143038810246, 0.23951558792401873876, 
  0.23951396560074672813, 0.23951316185458193696, 
  0.23951276896857760019, 0.23951256549076384140, 
  0.23951246457519508632, 0.23951241379894319892, 
  0.23951238831377561418, 0.23951237572189365026, 
  0.23951236987020473905, 0.23951236682783668844, 
  0.23951236526489169808, 0.23951236448039477211, 
  0.23951236408481104407, 0.23951236388903261005, 
  0.23951236379176348255, 0.23951236374127631960, 
  0.23951236371594470977, 0.23951236370315939552, 
  0.23951236369668972063, 0.23951236369348909216, 
  0.23951236369198337773, 0.23951236369120744022, 
  0.23951236369081708492, 0.23951236369062853799, 
  0.23951236369053620709, 0.23951236369048844531, 
  0.23951236369046493719, 0.23951236369045291280, 
  0.23951236369044699217, 0.23951236369044389681, 
  0.23951236369044233792, 0.23951236369044158182, 
  0.23951236369044121167], [0.45820740827425258366, 
  0.33344132130540262057, 0.29412954435573788140, 
  0.26601755712914654589, 0.25282741907074695497, 
  0.24624320090286375031, 0.24294791963156752699, 
  0.24131229120915330301, 0.24048949355684866117, 
  0.24007915617443445161, 0.23987183928137689125, 
  0.23976801604208514446, 0.23971646555560076331, 
  0.23969080965364659865, 0.23967772470163841417, 
  0.23967125261382119158, 0.23966799342297164239, 
  0.23966635355870716098, 0.23966553619859729138, 
  0.23966513159388069459, 0.23966492777471439766, 
  0.23966482598930950218, 0.23966477449737895073, 
  0.23966474861681203983, 0.23966473579451761850, 
  0.23966472959937629467, 0.23966472645037308076, 
  0.23966472485114883866, 0.23966472405084888986, 
  0.23966472365203260978, 0.23966472345263977158, 
  0.23966472335303797495, 0.23966472330213467238, 
  0.23966472327672879725, 0.23966472326398365548, 
  0.23966472325757653305, 0.23966472325438904647, 
  0.23966472325283649005, 0.23966472325204956495, 
  0.23966472325165419005, 0.23966472325146058726, 
  0.23966472325136533472, 0.23966472325131671385, 
  0.23966472325129249434, 0.23966472325128024849, 
  0.23966472325127416615, 0.23966472325127106812, 
  0.23966472325126951200, 0.23966472325126874677, 
  0.23966472325126836810], [0.46318218006807215074, 
  0.33478139615526386336, 0.29516526110032025506, 
  0.26657363199720195695, 0.25313067132351445217, 
  0.24643072380739099839, 0.24308230152317121917, 
  0.24141432309851926916, 0.24057803168600680781, 
  0.24016039975571866129, 0.23995041450229929943, 
  0.23984538960800159211, 0.23979304124345528144, 
  0.23976695291862330016, 0.23975377928593433371, 
  0.23974722624156057981, 0.23974393853243193863, 
  0.23974228621826863245, 0.23974146185752104822, 
  0.23974105280164318780, 0.23974084714190051754, 
  0.23974074380344866179, 0.23974069202610521375, 
  0.23974066608807244812, 0.23974065317682658918, 
  0.23974064682872220628, 0.23974064363355279618, 
  0.23974064202206995881, 0.23974064121422685224, 
  0.23974064081271112003, 0.23974064061147895968, 
  0.23974064051067816278, 0.23974064045993657581, 
  0.23974064043459179159, 0.23974064042185554359, 
  0.23974064041546204718, 0.23974064041228074784, 
  0.23974064041071159135, 0.23974064040991910790, 
  0.23974064040952314989, 0.23974064040932664404, 
  0.23974064040922907417, 0.23974064040917989165, 
  0.23974064040915527663, 0.23974064040914296540, 
  0.23974064040913682371, 0.23974064040913372258, 
  0.23974064040913216862, 0.23974064040913139941, 
  0.23974064040913101642], [0.46606277766293042327, 
  0.33553681938148991236, 0.29575341904933205323, 
  0.26688608270568623701, 0.25330031233057817134, 
  0.24653344884252361589, 0.24315402568582219387, 
  0.24146768588157199367, 0.24062340000895837982, 
  0.24020146501092003283, 0.23998992859364037061, 
  0.23988416835067906725, 0.23983136763948393136, 
  0.23980500200492696228, 0.23979175514609342415, 
  0.23978515079111925193, 0.23978184589816931929, 
  0.23978018865679912889, 0.23977936069183123634, 
  0.23977894801662044563, 0.23977874081492611506, 
  0.23977863693349404982, 0.23977858497702115498, 
  0.23977855900802838188, 0.23977854604634895478, 
  0.23977853962202480928, 0.23977853639792701602, 
  0.23977853477714356771, 0.23977853396589727165, 
  0.23977853356128036024, 0.23977853335910659610, 
  0.23977853325793879159, 0.23977853320710712482, 
  0.23977853318170118317, 0.23977853316897771725, 
  0.23977853316259732996, 0.23977853315941933789, 
  0.23977853315784034853, 0.23977853315704582328, 
  0.23977853315664871453, 0.23977853315645108917, 
  0.23977853315635240024, 0.23977853315630292952, 
  0.23977853315627821182, 0.23977853315626585275, 
  0.23977853315625967850, 0.23977853315625657860, 
  0.23977853315625502546, 0.23977853315625425253, 
  0.23977853315625386753], [0.46770187993902614739, 
  0.33595763591144558039, 0.29608199317900432986, 
  0.26705877563004852623, 0.25339359420106790865, 
  0.24658910988239190912, 0.24319208999285483523, 
  0.24149544285847017958, 0.24064658570990061072, 
  0.24022227231088118994, 0.24000983553533902062, 
  0.23990361461664421642, 0.23985054614866186090, 
  0.23982402855847706838, 0.23981073917509883370, 
  0.23980410480105900718, 0.23980078582112276087, 
  0.23979912405777883621, 0.23979829354764835074, 
  0.23979787890948667413, 0.23979767108404482732, 
  0.23979756702892524973, 0.23979751499645117234, 
  0.23979748899523771590, 0.23979747600915703026, 
  0.23979746954173245284, 0.23979746630089751062, 
  0.23979746467518947409, 0.23979746386246602370, 
  0.23979746345696299676, 0.23979746325400864505, 
  0.23979746315245752577, 0.23979746310161915252, 
  0.23979746307621631182, 0.23979746306348838099, 
  0.23979746305711247001, 0.23979746305393148386, 
  0.23979746305234610231, 0.23979746305155130138, 
  0.23979746305115390822, 0.23979746305095579035, 
  0.23979746305085671152, 0.23979746305080710349, 
  0.23979746305078229811, 0.23979746305076990191, 
  0.23979746305076370596, 0.23979746305076060013, 
  0.23979746305075904716, 0.23979746305075827330, 
  0.23979746305075788704]])

# epsmat = numpy.array([[10.6666666666666666667,4.666666666666666667,2.479166666666666667,1.177083333333333333,0.608072916666666667,0.300130208333333333,0.151285807291666667,0.075398763020833333,0.037775675455729167,0.018872578938802083,0.009441057840983073,0.004719575246175130,0.002360085646311442,0.001179983218510946,0.000590010235706965,0.000295001392563184,0.000147501860434810,0.000073750697386762,0.000036875421452957,0.000018437696174563,0.000009218852634755,0.000004609425407883,0.000002304712988159,0.000001152356437236,0.000000576178236381,0.000000288089114638,0.000000144044558429,0.000000072022278993,0.000000036011139566,0.000000018005569769,0.000000009002784889,0.000000004501392444,0.000000002250696222,0.000000001125348111,0.000000000562674056,0.000000000281337028,0.000000000140668514,0.000000000070334257,0.000000000035167128,0.000000000017583564,0.000000000008791782,0.000000000004395891,0.000000000002197946,0.000000000001098973,0.000000000000549486,0.000000000000274743,0.000000000000137372,0.000000000000068686,0.000000000000034343,0.000000000000017171],[29.489504777701372129,11.409961635658230085,5.432951146147740575,2.797679728968428659,1.416679369326216511,0.701049171116736846,0.352531661407146287,0.202683586711659482,0.089999972616607399,0.041868251093886209,0.024427325716614301,0.010774904779602762,0.005413183566200570,0.003242145552922700,0.001316380450814381,0.000730160296940550,0.000397248100800907,0.000180906998505109,0.000087988424553670,0.000048194811058064,0.000021221394821243,0.000010198320553667,0.000005823037886844,0.000002905463250935,0.000001290620689762,0.000000711026802848,0.000000315315988804,0.000000161054341598,0.000000098561385399,0.000000044626347375,0.000000020642678764,0.000000011811051576,0.000000005318269242,0.000000002726696301,0.000000001421618394,0.000000000692527547,0.000000000328742742,0.000000000185084193,0.000000000084897652,0.000000000043176152,0.000000000021883601,0.000000000009528193,0.000000000005053129,0.000000000002855370,0.000000000001191359,0.000000000000603697,0.000000000000364260,0.000000000000174703,0.000000000000081334,0.000000000000046098],[64.003284426382306780,23.469441778167611690,12.00037128091332293,6.36510420261271529,3.07448403145394857,1.59484033371151787,0.82766259676608501,0.46396447007253309,0.21430999972805044,0.10118550725872207,0.05384662162634353,0.02520484849343578,0.01298358946416007,0.00707119077034480,0.00313047600804245,0.00158610777858604,0.00091388263697388,0.00042280760749655,0.00021001377137831,0.00011465038229397,0.00005056172513568,0.00002497931079720,0.00001340959727046,0.00000681504800269,0.00000307500313929,0.00000157274173172,0.00000078710735701,0.00000038960204393,0.00000021111944785,0.00000009466351082,0.00000004527932748,0.00000002514849675,0.00000001299844827,0.00000000671077923,0.00000000311985788,0.00000000158747390,0.00000000074734678,0.00000000042681936,0.00000000020365898,0.00000000010134484,0.00000000005064424,0.00000000002309970,0.00000000001253214,0.00000000000661381,0.00000000000277240,0.00000000000139805,0.00000000000081194,0.00000000000040291,0.00000000000018381,0.00000000000009257],[130.073070109433878607,47.29464374043093092,25.93563481256528931,13.61185783801685906,6.82846840428810800,3.46584401251859252,1.76984671406146448,0.92351121793945300,0.44689578024463238,0.22832602672028930,0.11648466177898728,0.05445760550094023,0.02840322616169247,0.01533308146093830,0.00741089718480968,0.00358721550624493,0.00187109010588839,0.00088936706137918,0.00045954886073747,0.00023872082063362,0.00011033317893112,0.00005658156803517,0.00002902402253647,0.00001473544908067,0.00000678797609317,0.00000347605201686,0.00000175930242064,0.00000090322934672,0.00000045343104205,0.00000021648116162,0.00000010909715622,0.00000005623314951,0.00000002936593783,0.00000001460102108,0.00000000710480777,0.00000000359453379,0.00000000166979777,0.00000000089482410,0.00000000043805139,0.00000000022122631,0.00000000010885868,0.00000000005204877,0.00000000002784233,0.00000000001482984,0.00000000000646803,0.00000000000330734,0.00000000000179908,0.00000000000089867,0.00000000000041465,0.00000000000020653],[259.20267182347027876,94.27696897143444856,54.54988725250595337,28.46565925149863873,14.56716907448621926,7.39613557361651989,3.75179525207415430,1.91286730401455758,0.93223780060580905,0.47689402487513626,0.23846844554564236,0.11767064418376781,0.05964774040082835,0.03113892002054163,0.01560624678238143,0.00754558426379857,0.00389529100854837,0.00191231055640218,0.00097314285188133,0.00049110083406181,0.00023340466626779,0.00011882636324285,0.00006045969965344,0.00003056034950416,0.00001445352520962,0.00000727606030270,0.00000364566596098,0.00000189752503649,0.00000095500744497,0.00000045938632457,0.00000023315935897,0.00000011983184348,0.00000006090358977,0.00000003017931312,0.00000001490645029,0.00000000748398714,0.00000000354304260,0.00000000183889918,0.00000000092810943,0.00000000046580872,0.00000000022429756,0.00000000011182078,0.00000000005870911,0.00000000003031904,0.00000000001393158,0.00000000000717350,0.00000000000369134,0.00000000000184613,0.00000000000089399,0.00000000000043977],[514.54440127873186816,187.99374531619600176,112.56886947893328725,58.66195319154965812,30.26955320806802696,15.41844520658426633,7.80298040000761785,3.94877859468019694,1.95526393031876464,0.98995698692873887,0.49416073398635967,0.24529481622372054,0.12359695940391203,0.06323599801113533,0.03164294934212927,0.01558400393374778,0.00791333162687218,0.00391684180287019,0.00197815558795357,0.00099846593806697,0.00048613925458710,0.00024451962476591,0.00012309642663604,0.00006250912009652,0.00003006532124215,0.00001520352013017,0.00000761228038602,0.00000390608798429,0.00000195314700877,0.00000095591436312,0.00000048053730836,0.00000024493551455,0.00000012398460542,0.00000006067560725,0.00000003038453641,0.00000001526885260,0.00000000735539561,0.00000000375399923,0.00000000188831835,0.00000000095153577,0.00000000046414545,0.00000000023257852,0.00000000011887755,0.00000000006069912,0.00000000002887220,0.00000000001476132,0.00000000000743097,0.00000000000373039,0.00000000000183279,0.00000000000090791],[1022.16148051879972743,374.9830563772419649,229.4248043477622516,119.5529723922003344,61.9460961079540105,31.6003132272385804,15.9841014761324190,8.0614202814248999,4.0124943851024541,2.0208279699457394,1.0112516873352072,0.5039790317053261,0.2521864494159917,0.1276511080647678,0.0637385437348844,0.0317712484706861,0.0160910394369913,0.0079689806811657,0.0040099632094903,0.0020075964537296,0.0009920791651935,0.0004968935773969,0.0002500366865478,0.0001260090834141,0.0000618007616561,0.0000310008768600,0.0000155287378249,0.0000078537250321,0.0000039269702376,0.0000019423212692,0.0000009784420067,0.0000004940652521,0.0000002479486602,0.0000001220823437,0.0000000611854353,0.0000000307090554,0.0000000151119827,0.0000000076501445,0.0000000038408976,0.0000000019273126,0.0000000009455922,0.0000000004744204,0.0000000002395270,0.0000000001211703,0.0000000000592389,0.0000000000298361,0.0000000000149721,0.0000000000075042,0.0000000000037296,0.0000000000018513],[2034.4168582341634018,748.6131934374390598,463.9121137628992428,241.7761209679317356,125.5644196032795792,64.0956398398607867,32.4180209254823545,16.3244564515244970,8.1543748658966762,4.0950583635207193,2.0488706060908003,1.0231016515906234,0.5117985714640848,0.2576251336910928,0.1285550237924693,0.0641193361050193,0.0322951733903170,0.0160856910698791,0.0080713697552242,0.0040414199168555,0.0020067457579411,0.0010032255754327,0.0005035947397358,0.0002526656573758,0.0001252770237923,0.0000627524928586,0.0000314455064558,0.0000157904617525,0.0000078936798242,0.0000039156996634,0.0000019687745038,0.0000009904039855,0.0000004958785292,0.0000002460016520,0.0000001234184003,0.0000000616946010,0.0000000306184428,0.0000000154191154,0.0000000077376719,0.0000000038751643,0.0000000019180608,0.0000000009597916,0.0000000004825440,0.0000000002420160,0.0000000001196181,0.0000000000600587,0.0000000000300920,0.0000000000150539,0.0000000000075204,0.0000000000037527],[4055.8844501875043031,1495.4959270703569129,933.6439633120189142,486.6626018155520543,253.0655123049827486,129.2242567144691625,65.3523759874308175,32.8864324574635387,16.4592377450695034,8.2536566159765036,4.1296824493329107,2.0640346247375763,1.0322978629674841,0.5179785618775505,0.2588313739911675,0.1292368285120104,0.0648114343770433,0.0323369113078067,0.0162080555357858,0.0081191999714603,0.0040429745153504,0.0020220367566729,0.0010115174765534,0.0005070058036090,0.0002524794818205,0.0001262549088379,0.0000632089812399,0.0000316993025866,0.0000158434329041,0.0000078901071275,0.0000039539666506,0.0000019846533034,0.0000009934888287,0.0000004944039646,0.0000002477439579,0.0000001237058572,0.0000000616494982,0.0000000309799919,0.0000000155037673,0.0000000077586591,0.0000000038589161,0.0000000019314466,0.0000000009676123,0.0000000004845282,0.0000000002411694,0.0000000001207248,0.0000000000603982,0.0000000000302113,0.0000000000151042,0.0000000000075444],[8095.7882732196243279,2988.902480937151044,1873.884012048167360,976.886341902362142,508.331685360860602,259.618235484666866,131.292051938272812,66.046496873152533,33.087730980368595,16.580251499686119,8.296028686876712,4.147942663867931,2.074340822267021,1.039119917346248,0.519383559098326,0.259525151310981,0.129957222322200,0.064901210003188,0.032481814904116,0.016259654837293,0.008114189998569,0.004059983879752,0.002029827842055,0.001015836112168,0.000507019486445,0.000253427849347,0.000126796500244,0.000063513440673,0.000031735918977,0.000015842031188,0.000007927447733,0.000003972203480,0.000001987962285,0.000000991474766,0.000000496291205,0.000000248035198,0.000000123810029,0.000000062035937,0.000000031032910,0.000000015514449,0.000000007740044,0.000000003874737,0.000000001939495,0.000000000970354,0.000000000484042,0.000000000242075,0.000000000121044,0.000000000060536,0.000000000030277,0.000000000015127]])

epsmat = numpy.array(
[[10.666666666666666667,4.666666666666666667,2.479166666666666667,1.177083333333333333,0.608072916666666667,0.300130208333333333,0.151285807291666667,0.075398763020833333,0.037775675455729167,0.018872578938802083,0.009441057840983073,0.004719575246175130,0.002360085646311442,0.001179983218510946,0.000590010235706965,0.000295001392563184,0.000147501860434810,0.000073750697386762,0.000036875421452957,0.000018437696174563,0.000009218852634755,0.000004609425407883,0.000002304712988159,0.000001152356437236,0.000000576178236381,0.000000288089114638,0.000000144044558429,0.000000072022278993,0.000000036011139566,0.000000018005569769,0.000000009002784889,0.000000004501392444,0.000000002250696222,0.000000001125348111,0.000000000562674056,0.000000000281337028,0.000000000140668514,0.000000000070334257,0.000000000035167128,0.000000000017583564,0.000000000008791782,0.000000000004395891,0.000000000002197946,0.000000000001098973,0.000000000000549486,0.000000000000274743,0.000000000000137372,0.000000000000068686,0.000000000000034343,0.000000000000017171],[29.489504777701372129,11.409961635658230085,5.432951146147740575,2.797679728968428659,1.416679369326216511,0.701049171116736846,0.352531661407146287,0.202683586711659482,0.089999972616607399,0.041868251093886209,0.024427325716614301,0.010774904779602762,0.005413183566200570,0.003242145552922700,0.001316380450814381,0.000730160296940550,0.000397248100800907,0.000180906998505109,0.000087988424553670,0.000048194811058064,0.000021221394821243,0.000010198320553667,0.000005823037886844,0.000002905463250935,0.000001290620689762,0.000000711026802848,0.000000315315988804,0.000000161054341598,0.000000098561385399,0.000000044626347375,0.000000020642678764,0.000000011811051576,0.000000005318269242,0.000000002726696301,0.000000001421618394,0.000000000692527547,0.000000000328742742,0.000000000185084193,0.000000000084897652,0.000000000043176152,0.000000000021883601,0.000000000009528193,0.000000000005053129,0.000000000002855370,0.000000000001191359,0.000000000000603697,0.000000000000364260,0.000000000000174703,0.000000000000081334,0.000000000000046098],[64.003284426382306780,23.469441778167611690,12.00037128091332293,6.36510420261271529,3.07448403145394857,1.59484033371151787,0.82766259676608501,0.46396447007253309,0.21430999972805044,0.10118550725872207,0.05384662162634353,0.02520484849343578,0.01298358946416007,0.00707119077034480,0.00313047600804245,0.00158610777858604,0.00091388263697388,0.00042280760749655,0.00021001377137831,0.00011465038229397,0.00005056172513568,0.00002497931079720,0.00001340959727046,0.00000681504800269,0.00000307500313929,0.00000157274173172,0.00000078710735701,0.00000038960204393,0.00000021111944785,0.00000009466351082,0.00000004527932748,0.00000002514849675,0.00000001299844827,0.00000000671077923,0.00000000311985788,0.00000000158747390,0.00000000074734678,0.00000000042681936,0.00000000020365898,0.00000000010134484,0.00000000005064424,0.00000000002309970,0.00000000001253214,0.00000000000661381,0.00000000000277240,0.00000000000139805,0.00000000000081194,0.00000000000040291,0.00000000000018381,0.00000000000009257],[130.07307010943387861,47.29464374043093092,25.93563481256528931,13.61185783801685906,6.82846840428810800,3.46584401251859252,1.76984671406146448,0.92351121793945300,0.44689578024463238,0.22832602672028930,0.11648466177898728,0.05445760550094023,0.02840322616169247,0.01533308146093830,0.00741089718480968,0.00358721550624493,0.00187109010588839,0.00088936706137918,0.00045954886073747,0.00023872082063362,0.00011033317893112,0.00005658156803517,0.00002902402253647,0.00001473544908067,0.00000678797609317,0.00000347605201686,0.00000175930242064,0.00000090322934672,0.00000045343104205,0.00000021648116162,0.00000010909715622,0.00000005623314951,0.00000002936593783,0.00000001460102108,0.00000000710480777,0.00000000359453379,0.00000000166979777,0.00000000089482410,0.00000000043805139,0.00000000022122631,0.00000000010885868,0.00000000005204877,0.00000000002784233,0.00000000001482984,0.00000000000646803,0.00000000000330734,0.00000000000179908,0.00000000000089867,0.00000000000041465,0.00000000000020653],[259.20267182347027876,94.27696897143444856,54.54988725250595337,28.46565925149863873,14.56716907448621926,7.39613557361651989,3.75179525207415430,1.91286730401455758,0.93223780060580905,0.47689402487513626,0.23846844554564236,0.11767064418376781,0.05964774040082835,0.03113892002054163,0.01560624678238143,0.00754558426379857,0.00389529100854837,0.00191231055640218,0.00097314285188133,0.00049110083406181,0.00023340466626779,0.00011882636324285,0.00006045969965344,0.00003056034950416,0.00001445352520962,0.00000727606030270,0.00000364566596098,0.00000189752503649,0.00000095500744497,0.00000045938632457,0.00000023315935897,0.00000011983184348,0.00000006090358977,0.00000003017931312,0.00000001490645029,0.00000000748398714,0.00000000354304260,0.00000000183889918,0.00000000092810943,0.00000000046580872,0.00000000022429756,0.00000000011182078,0.00000000005870911,0.00000000003031904,0.00000000001393158,0.00000000000717350,0.00000000000369134,0.00000000000184613,0.00000000000089399,0.00000000000043977],[514.54440127873186816,187.99374531619600176,112.56886947893328725,58.66195319154965812,30.26955320806802696,15.41844520658426633,7.80298040000761785,3.94877859468019694,1.95526393031876464,0.98995698692873887,0.49416073398635967,0.24529481622372054,0.12359695940391203,0.06323599801113533,0.03164294934212927,0.01558400393374778,0.00791333162687218,0.00391684180287019,0.00197815558795357,0.00099846593806697,0.00048613925458710,0.00024451962476591,0.00012309642663604,0.00006250912009652,0.00003006532124215,0.00001520352013017,0.00000761228038602,0.00000390608798429,0.00000195314700877,0.00000095591436312,0.00000048053730836,0.00000024493551455,0.00000012398460542,0.00000006067560725,0.00000003038453641,0.00000001526885260,0.00000000735539561,0.00000000375399923,0.00000000188831835,0.00000000095153577,0.00000000046414545,0.00000000023257852,0.00000000011887755,0.00000000006069912,0.00000000002887220,0.00000000001476132,0.00000000000743097,0.00000000000373039,0.00000000000183279,0.00000000000090791],[1022.1614805187997274,374.9830563772419649,229.4248043477622516,119.5529723922003344,61.9460961079540105,31.6003132272385804,15.9841014761324190,8.0614202814248999,4.0124943851024541,2.0208279699457394,1.0112516873352072,0.5039790317053261,0.2521864494159917,0.1276511080647678,0.0637385437348844,0.0317712484706861,0.0160910394369913,0.0079689806811657,0.0040099632094903,0.0020075964537296,0.0009920791651935,0.0004968935773969,0.0002500366865478,0.0001260090834141,0.0000618007616561,0.0000310008768600,0.0000155287378249,0.0000078537250321,0.0000039269702376,0.0000019423212692,0.0000009784420067,0.0000004940652521,0.0000002479486602,0.0000001220823437,0.0000000611854353,0.0000000307090554,0.0000000151119827,0.0000000076501445,0.0000000038408976,0.0000000019273126,0.0000000009455922,0.0000000004744204,0.0000000002395270,0.0000000001211703,0.0000000000592389,0.0000000000298361,0.0000000000149721,0.0000000000075042,0.0000000000037296,0.0000000000018513],[2034.4168582341634018,748.6131934374390598,463.9121137628992428,241.7761209679317356,125.5644196032795792,64.0956398398607867,32.4180209254823545,16.3244564515244970,8.1543748658966762,4.0950583635207193,2.0488706060908003,1.0231016515906234,0.5117985714640848,0.2576251336910928,0.1285550237924693,0.0641193361050193,0.0322951733903170,0.0160856910698791,0.0080713697552242,0.0040414199168555,0.0020067457579411,0.0010032255754327,0.0005035947397358,0.0002526656573758,0.0001252770237923,0.0000627524928586,0.0000314455064558,0.0000157904617525,0.0000078936798242,0.0000039156996634,0.0000019687745038,0.0000009904039855,0.0000004958785292,0.0000002460016520,0.0000001234184003,0.0000000616946010,0.0000000306184428,0.0000000154191154,0.0000000077376719,0.0000000038751643,0.0000000019180608,0.0000000009597916,0.0000000004825440,0.0000000002420160,0.0000000001196181,0.0000000000600587,0.0000000000300920,0.0000000000150539,0.0000000000075204,0.0000000000037527],[4055.8844501875043031,1495.4959270703569129,933.6439633120189142,486.6626018155520543,253.0655123049827486,129.2242567144691625,65.3523759874308175,32.8864324574635387,16.4592377450695034,8.2536566159765036,4.1296824493329107,2.0640346247375763,1.0322978629674841,0.5179785618775505,0.2588313739911675,0.1292368285120104,0.0648114343770433,0.0323369113078067,0.0162080555357858,0.0081191999714603,0.0040429745153504,0.0020220367566729,0.0010115174765534,0.0005070058036090,0.0002524794818205,0.0001262549088379,0.0000632089812399,0.0000316993025866,0.0000158434329041,0.0000078901071275,0.0000039539666506,0.0000019846533034,0.0000009934888287,0.0000004944039646,0.0000002477439579,0.0000001237058572,0.0000000616494982,0.0000000309799919,0.0000000155037673,0.0000000077586591,0.0000000038589161,0.0000000019314466,0.0000000009676123,0.0000000004845282,0.0000000002411694,0.0000000001207248,0.0000000000603982,0.0000000000302113,0.0000000000151042,0.0000000000075444],[8095.7882732196243279,2988.902480937151044,1873.884012048167360,976.886341902362142,508.331685360860602,259.618235484666866,131.292051938272812,66.046496873152533,33.087730980368595,16.580251499686119,8.296028686876712,4.147942663867931,2.074340822267021,1.039119917346248,0.519383559098326,0.259525151310981,0.129957222322200,0.064901210003188,0.032481814904116,0.016259654837293,0.008114189998569,0.004059983879752,0.002029827842055,0.001015836112168,0.000507019486445,0.000253427849347,0.000126796500244,0.000063513440673,0.000031735918977,0.000015842031188,0.000007927447733,0.000003972203480,0.000001987962285,0.000000991474766,0.000000496291205,0.000000248035198,0.000000123810029,0.000000062035937,0.000000031032910,0.000000015514449,0.000000007740044,0.000000003874737,0.000000001939495,0.000000000970354,0.000000000484042,0.000000000242075,0.000000000121044,0.000000000060536,0.000000000030277,0.000000000015127]]
)