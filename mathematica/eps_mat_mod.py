import math

eps_mat_string = """{{10.66666666666666666666666666666666666664`20.425968732272285, 
   4.66666666666666666666666666666666666664`19.669006780958572, 
   2.47916666666666666666666666666666666664`19.30839109531035, 
   1.17708333333333333333333333333333333332`18.937467931809124, 
   0.60807291666666666666666666666666666664`18.63601377919354, 
   0.30013020833333333333333333333333333332`18.318817646019788, 
   0.15128580729166666666666666666666666664`18.01796975160469, 
   0.07539876302083333333333333333333333332`17.712968519990742, 
   0.03777567545572916666666666666666666664`17.412000834842917, 
   0.01887257893880208333333333333333333332`17.10998227225979, 
   0.00944105784098307291666666666666666664`16.80896888565609, 
   0.00471957524617513020833333333333333332`16.507692011110606, 
   0.00236008564631144205729166666666666664`16.206666231905775, 
   0.00117998321851094563802083333333333332`15.905574532914121, 
   0.00059001023570696512858072916666666664`15.604545595372855, 
   0.00029500139256318410237630208333333332`15.30350017490095, 
   0.00014750186043481032053629557291666664`15.002470444018075, 
   0.00007375069738676150639851888020833332`14.701436592216107, 
   0.00003687542145295689503351847330729164`14.400406662763052, 
   0.00001843769617456321914990743001302082`14.099375703068574, 
   9.2188526347551184395949045817057`13.7983457239583*^-6, 
   4.60942540788285744686921437581379`13.497315487286949*^-6, 
   2.30471298815852302747468153635658`13.196285495761453*^-6, 
   1.15235643723584265292932589848835`12.895255439845645*^-6, 
   5.7617823638148972046716759602226`12.594225445216287*^-7, 
   2.8808911463803118143308286865551`12.293195434489352*^-7, 
   1.4404455842923861534169797475137`11.99216543908403*^-7, 
   7.202227899257470274581767929096`11.691135439654307*^-8, 
   3.601113956567629041198112342194`11.390105444054988*^-8, 
   1.800556976896035739817610495568`11.089075447449574*^-8, 
   9.00278488881698738903007021385`10.788045451801759*^-9, 
   4.50139244354113195652663155972`10.487015455902421*^-9, 
   2.25069622204161652138469188834`10.185985460242478*^-9, 
   1.12534811096659815206807072247`9.884955464519658*^-9, 
   5.62674055500239734979121368`9.583925468856688*^-10, 
   2.8133702774673173570054348265`9.282895473177994*^-10, 
   1.4066851387442465903433961673`8.981865477514267*^-10, 
   7.033425693700057128035623328`8.680835481846607*^-11, 
   3.516712846856646008918235883`8.37980548618269*^-11, 
   1.758356423426999515479033098`8.078775490517787*^-11, 
   8.79178211713913348045793061`7.777745494853823*^-12, 
   4.39589105856873955961641228`7.476715499189613*^-12, 
   2.19794552928462827374962894`7.175685503525633*^-12, 
   1.09897276464226243808652991`6.874655507861597*^-12, 
   5.4948638232114737491460386`6.573625512197613*^-13, 
   2.7474319116057045628303415`6.272595516533619*^-13, 
   1.3737159558028623788347574`5.971565520869638*^-13, 
   6.868579779014291699334613`5.670535525205657*^-14, 
   3.434289889507152160554546`5.369505529541671*^-14, 
   1.717144944753574818099825`5.06847553387769*^-14}, \
{29.48950477770137212861671066005431330473`20.277588114991712, 
   11.40996163565823008547466751801117126177`19.521041477330144, 
   5.43295114614774057498515702850068175136`19.12624242582798, 
   2.7976797289684286588311356113213698351`18.815148163684704, 
   1.41667936932621651081403403384827533417`18.506863275018514, 
   0.70104917111673684574226369892004566921`18.194990389993585, 
   0.35253166140714628705631027612451761052`17.895613111139298, 
   0.20268358671165948219450541431965580561`17.65412724693427, 
   0.08999997261660739886117208098632247225`17.30102419396026, 
   0.04186825109388620909337337355913207309`16.968367488090205, 
   0.02442732571661430090715854671930523336`16.734235339592217, 
   0.01077490477960276197453807113007964391`16.37875605487715, 
   0.00541318356620057031623302386593311495`16.07978230031944, 
   0.00324214555292270031839232290023214935`15.85713111240894, 
   0.00131638045081438071796914581689881607`15.465674266758155, 
   0.0007301602969405503491827604899861454`15.20970729399839, 
   0.00039724810080090659825922393336243059`14.945349138459445, 
   0.00018090699850510863259558431397374825`14.603741975448886, 
   0.00008798842455367020675674618384724234`14.290711871718438, 
   0.00004819481105806367503570176515633113`14.029286422888653, 
   0.00002122139482124291564785886603101599`13.673059939541224, 
   0.00001019832055366668615778572450866478`13.354814577143504, 
   5.82303788684398369187751115710228`13.111435513574639*^-6, 
   2.90546325093522277133442010628502`12.809501279298896*^-6, 
   1.29062068976185478689314135692963`12.4570845128941*^-6, 
   7.1102680284834729951917427212958`12.198171854591962*^-7, 
   3.1531598880351377883562614113149`11.845031873599883*^-7, 
   1.6105434159801784004735965458826`11.553258316241797*^-7, 
   9.856138539894317582112750113727`11.339992678211969*^-8, 
   4.462634737523244854776388198707`10.995877220343859*^-8, 
   2.064267876413950204933526377111`10.661051932866537*^-8, 
   1.181105157631240773896835421488`10.418574444542914*^-8, 
   5.31826924157288137006075005347`10.072056198589278*^-9, 
   2.72669630097330200072393516842`9.7819226474963*^-9, 
   1.4216183939855640783858642867`9.499068912574751*^-9, 
   6.9252754703740435849740091763`9.186722931748497*^-10, 
   3.2874274197547053469379754405`8.863142051698912*^-10, 
   1.8508419265127484591603980212`8.613655207372299*^-10, 
   8.489765202228367125495635959`8.275181557817746*^-11, 
   4.317615156170624798694838282`7.981529808071025*^-11, 
   2.188360056198774367710489141`7.686404657569605*^-11, 
   9.52819347154717117442755115`7.325296445488163*^-12, 
   5.0531293403977382003979393`7.0498462931274295*^-12, 
   2.85537015604633234672396069`6.801948294519721*^-12, 
   1.19135879629254288066135414`6.422328454244943*^-12, 
   6.0369700253890759489353174`6.127104898013288*^-13, 
   3.6426008489173398710601528`5.907697462972469*^-13, 
   1.7470269158225775704346911`5.5885854745575925*^-13, 
   8.133403855388397248890543`5.256558215761655*^-14, 
   4.609816367093620166374225`5.009969504033871*^-14}, \
{64.00328442638230677958147841129069810706`20.179412944150506, 
   23.46944177816761169023167892129811054505`19.43464532172322, 
   12.00037128091332292921730768410502891636`19.07808914926397, 
   6.3651042026127152897914660825467494474`18.788671593162775, 
   3.07448403145394856653157639417408830531`18.46527545553483, 
   1.59484033371151786748589744067096548162`18.177814204373018, 
   0.82766259676608500773920783073218634678`17.8924133699909, 
   0.46396447007253308983227439599221159098`17.640113157502732, 
   0.21430999972805044032082087123922404744`17.30448942839384, 
   0.10118550725872206504383217962799260189`16.978522341426306, 
   0.05384662162634352807923401318549840428`16.70458152571578, 
   0.02520484849343578093386484183480139636`16.374916113014766, 
   0.01298358946416007378811331522192287862`16.086819978195148, 
   0.00707119077034480147778321663868957337`15.822905476517118, 
   0.00313047600804244563787514993898804744`15.469022555269808, 
   0.00158610777858603560378060445744298833`15.173742690156482, 
   0.00091388263697387607108464231165802038`14.934299697456192, 
   0.00042280760749655187576055389788410048`14.599551988616668, 
   0.00021001377137830591664323118338825086`14.295657015682956, 
   0.0001146503822939702457745493744051412`14.032784696433875, 
   0.0000505617251356827328460207692407258`13.677231070497928, 
   0.00002497931079719879724323177283906733`13.370989635818244, 
   0.00001340959727046159532707845319629509`13.100824919560615, 
   6.81504800268841960174232398201107`12.806878110392825*^-6, 
   3.07500313929005748086361560544003`12.461254755724918*^-6, 
   1.57274173171508500953857207879612`12.170066599214694*^-6, 
   7.8710735700758667270768229127162`11.869443159041634*^-7, 
   3.8960204392704332981032792364166`11.564030413421891*^-7, 
   2.1111944784629814565706960138706`11.297937427850426*^-7, 
   9.466351082268880703801191036429`10.949591793707288*^-8, 
   4.527932747982256676412294089958`10.629309154237868*^-8, 
   2.514849675317128283396349608394`10.373921216729125*^-8, 
   1.299844826525356726634514249567`10.087300696410361*^-8, 
   6.71077923054390029095905146795`9.800182138155453*^-9, 
   3.11985788499785982970877471388`9.467543998002027*^-9, 
   1.58747389921643010997801439679`9.174115779888469*^-9, 
   7.4734678499555916426972834721`8.846931356983164*^-10, 
   4.2681935691305223057294180781`8.603653293529819*^-10, 
   2.0365897854504756809715828929`8.282312747635244*^-10, 
   1.0134483853430649670584344241`7.979210821499098*^-10, 
   5.064423728518864417433065179`7.677939221345012*^-11, 
   2.309970410336495591789293112`7.337015603223631*^-11, 
   1.253214456078434918545345343`7.071434582321284*^-11, 
   6.61381039557453087070252965`6.793860926830381*^-12, 
   2.7723974030497538612456085`6.416264669773802*^-12, 
   1.39804627098786806137469863`6.118930731854878*^-12, 
   8.1194149694278018905670576`5.882933924430993*^-13, 
   4.0291380254817587664474448`5.578621331603213*^-13, 
   1.8380689006015552828809237`5.237770973432339*^-13, 
   9.256648408639470431453817`4.939862954828306*^-14}, \
{130.07307010943387860715756828790533548733`20.11773044690176, 
   47.29464374043093091830642740556307366665`19.39159467735247, 
   25.93563481256528931300020944615422296305`19.068607408783347, 
   13.61185783801685906139623542689350384529`18.77916049189834, 
   6.82846840428810800484988939601937289595`18.47573159234039, 
   3.46584401251859252047740243102333846763`18.18022120898542, 
   1.76984671406146447610714649958984700492`17.88812000356834, 
   0.92351121793945299725960734040127282836`17.60541198225141, 
   0.44689578024463238084936496845691827394`17.290185953662945, 
   0.22832602672028929877882675839362414907`16.998571041542295, 
   0.1164846617789872842695050692850171604`16.706328335811936, 
   0.05445760550094023381901173792591283151`16.376144861296908, 
   0.02840322616169247138000061617949331267`16.093455112477525, 
   0.01533308146093830048259299383261118951`15.82571528888632, 
   0.00741089718480967657206971687610085034`15.509957998320791, 
   0.00358721550624492785061400890063511786`15.19484472208013, 
   0.00187109010588839193110478498806702805`14.91218218383238, 
   0.00088936706137918140921534220332885555`14.58916873690242, 
   0.00045954886073747063451592782832841733`14.302419583981441, 
   0.00023872082063362443493785066160751639`14.017978224535222, 
   0.00011033317893112021580489673230267461`13.682794095634506, 
   0.00005658156803516900724777427283425743`13.392762958608783, 
   0.00002902402253647102142776366357266375`13.102845591936452, 
   0.00001473544908066986285640391862705848`12.808451374194561, 
   6.78797609316699487150542148342482`12.471828304331742*^-6, 
   3.47605201686067126128970650485231`12.181174265742888*^-6, 
   1.75930242063516391317806621817851`11.885428498879195*^-6, 
   9.0322934672313843297002373052571`11.595886038393225*^-7, 
   4.5343104205266901203029991196132`11.296599247378253*^-7, 
   2.1648116161645574322773678681031`10.975508108291539*^-7, 
   1.0909715622460783828381447162942`10.677901428902523*^-7, 
   5.623314950709883974005858971997`10.39008040684531*^-8, 
   2.936593783388242120010707911084`10.107931873762583*^-8, 
   1.460102108016436010789083095479`9.80447122666371*^-8, 
   7.10480777022920804322054054961`9.491640330714942*^-9, 
   3.59453379494575645740504473317`9.195730569793616*^-9, 
   1.66979776785515447384113858488`8.862751874859175*^-9, 
   8.9482409676777162973929118368`8.591825669417664*^-10, 
   4.3805139373637254046110989486`8.281613065151094*^-10, 
   2.2122631111584246841424478911`7.9849247763594375*^-10, 
   1.0885868428689606019237565304`7.676951079637655*^-10, 
   5.204877408055107121037838276`7.356498503602332*^-11, 
   2.784232619631671163656068782`7.084793515985484*^-11, 
   1.482984319975743471074492599`6.811224557826526*^-11, 
   6.46802778745896745491520318`6.450859875717621*^-12, 
   3.30734329339681108457899597`6.15956727455585*^-12, 
   1.79908272500533497581834234`5.895139132128659*^-12, 
   8.986740817202572292513477`5.593690215263135*^-13, 
   4.1465485114450850069156544`5.257774749356883*^-13, 
   2.0652802217868613558503236`4.955066984721586*^-13}, \
{259.20267182347027876429008487106538637953`20.078830474796796, 
   94.27696897143444856467963084837537683034`19.36640115625975, 
   54.54988725250595336604173893308383538444`19.06824187554306, 
   28.46565925149863873011332347533534216525`18.779101803366473, 
   14.56716907448621926175670085883608432575`18.48639490392912, 
   7.39613557361651989443926178581524920713`18.192074453482352, 
   3.75179525207415429643264139345898633286`17.897566721561834, 
   1.9128673040145575750493527893475736609`17.6051161472076, 
   0.93223780060580904746626734863442686108`17.293071211085845, 
   0.4768940248751362599607276199055313108`17.00202151349675, 
   0.23846844554564236321815294374180772777`16.701075252338363, 
   0.11767064418376780911540000771081198258`16.394339855166223, 
   0.05964774040082834843665934911669636721`16.0992733843435, 
   0.03113892002054162800947993840873267809`15.816985337470973, 
   0.01560624678238142877350164583377639437`15.516983124881671, 
   0.00754558426379857424507589875591238255`15.201378512392235, 
   0.0038952910085483735897253216097080281`14.914226136305235, 
   0.00191231055640218138905351175662261513`14.605245104577227, 
   0.0009731428518813259477927149643363407`14.311863498734995, 
   0.00049110083406180876836284285054574741`14.014857639825703, 
   0.0002334046662677894204496876599866545`13.691796555134431, 
   0.00011882636324285214225593102673458534`13.398599849367978, 
   0.00006045969965343745129426748209592739`13.105153043489787, 
   0.00003056034950415762849578453460968289`12.80884538298171, 
   0.00001445352520961730972740917803691774`12.483660853249589, 
   7.27606030269551695716890020510332`12.185583358703344*^-6, 
   3.64566596098137340996557227180507`11.885463942546702*^-6, 
   1.89752503649189934514048667895275`11.601874584270872*^-6, 
   9.5500744497056725497023334846253`11.303693826825146*^-7, 
   4.5938632456974736611186501589856`10.985865132248005*^-7, 
   2.3315935897197517815422699398444`10.691339922346803*^-7, 
   1.1983184347557884825137575930282`10.40225931021744*^-7, 
   6.090358977404523521510283283948`10.108329961272663*^-8, 
   3.017931311862247176178950931249`9.803396420751474*^-8, 
   1.490645028915272111556252406401`9.497061305866366*^-8, 
   7.48398713661771539943788774284`9.197820102120184*^-9, 
   3.5430426046221899652920982201`8.87306344445129*^-9, 
   1.8388991838655336716401583474`8.58824498980598*^-9, 
   9.2810943467973822055179323787`8.291286257260106*^-10, 
   4.6580872487387596507085898621`7.991894688621345*^-10, 
   2.2429755610183487172154595872`7.674511611382151*^-10, 
   1.1182077661999636711537198336`7.3722095739573685*^-10, 
   5.870910584208696452717872203`7.092392535741334*^-11, 
   3.031904232183896052215646784`6.805402549005049*^-11, 
   1.393158135946751323172814458`6.467687485294937*^-11, 
   7.17350343274027450230971868`6.179418380218991*^-12, 
   3.69133975687558045234454211`5.8908710899543415*^-12, 
   1.84613317447786150749046998`5.589950096266826*^-12, 
   8.9398647280331364342876197`5.275018017149366*^-13, 
   4.3976727731960277777660491`4.966909980525431*^-13}, \
{514.54440127873186815860020015312615350953`20.054980723866645, 
   187.99374531619600176121777745768879832628`19.352787676781098, 
   112.56886947893328724676561895287613566508`19.070196077105603, 
   58.66195319154965811789855979408428649179`18.78227518222929, 
   30.26955320806802696364340151906381039077`18.494427460881855, 
   15.41844520658426633281249454465362544256`18.202052471663222, 
   7.8029804000076178475339893587843851805`17.906790619242415, 
   3.94877859468019693981122852579293202386`17.611260461555197, 
   1.95526393031876463984170334875738403433`17.30617305523031, 
   0.98995698692873886927531032746159681919`17.010667440795793, 
   0.49416073398635967017258114062102500766`16.708969074797178, 
   0.24529481622372053587531979384113668036`16.404816176963305, 
   0.12359695940391202771601845556074238285`16.107147406774263, 
   0.06323599801113533278868685817485521689`15.816108969386551, 
   0.03164294934212927072147603956549523567`15.51542512241746, 
   0.01558400393374778273969563687139773427`15.207828856285438, 
   0.00791333162687218000804531091801858105`14.913510029935482, 
   0.00391684180287019288394028190275036838`14.608087151457921, 
   0.0019781555879535655807777421640086272`14.311411782290623, 
   0.00099846593806697383401654833117803833`14.014484680056544, 
   0.00048613925458710242144782115092302564`13.701912175294265, 
   0.0002445196247659081647906474959170505`13.403465232886878, 
   0.00012309642663603748555307390553860622`13.105396972626025, 
   0.00006250912009651842787071791479326907`12.811094919662887, 
   0.00003006532124214953826052260470019296`12.493217385997143, 
   0.00001520352013017162553941498426339942`12.19709569181965, 
   7.61228038601564399783406351634989`11.896666315447263*^-6, 
   3.90608798429331822069724006972854`11.606893560919088*^-6, 
   1.95314700876528049469946334720222`11.305886472459033*^-6, 
   9.5591436312460218065863027651129`10.995570526893971*^-7, 
   4.8053730836237054114092675320094`10.696878651218555*^-7, 
   2.4493551454525608328296734239592`10.404203300239411*^-7, 
   1.2398460541808522174945553270253`10.108519304040747*^-7, 
   6.067560725450881862548446398449`9.798165671369063*^-8, 
   3.038453641442461168794009125109`9.49780415441349*^-8, 
   1.526885260240541391022102400029`9.198957942501476*^-8, 
   7.35539561202541923511040383955`8.881757576180965*^-9, 
   3.75399922821412057309330897209`8.589645718837101*^-9, 
   1.88831834882816757201103869976`8.291226753044667*^-9, 
   9.5153577217971756061923393366`7.993576659701227*^-10, 
   4.6414544990843280455448344463`7.6818056371913155*^-10, 
   2.3257852319174981812199874952`7.381721148382885*^-10, 
   1.188775505951559988069040875`7.090251387954885*^-10, 
   6.069912319996679089299371885`6.798333957567127*^-11, 
   2.887219952819577515514773045`6.475631410195968*^-11, 
   1.476131934800582510213766608`6.184276715750849*^-11, 
   7.43097462832941131500137862`5.8861973183298195*^-12, 
   3.73039375773443565546937359`5.586906215558001*^-12, 
   1.83278554134611005475077152`5.278263189929935*^-12, 
   9.0790721246165795338902632`4.973193006008705*^-13}, \
{1022.16148051879972743465383733420854758`20.040533700662433, 
   374.98305637724196493441533986961294741238`19.345128514058302, 
   229.4248043477622515633930928129224037832`19.07229465519191, 
   119.55297239220033440337074969073545265128`18.785418362865535, 
   61.9460961079540104526733064816432388838`18.500056214734087, 
   31.60031322723858039240539245399605673553`18.20864340123962, 
   15.98410147613241901597563784784397167124`17.913310792700795, 
   8.06142028142489990686505917334338109035`17.61640060416447, 
   4.01249438510245406697296343165533801`17.313608849544902, 
   2.02082796994573942889504758116558754005`17.015827833326107, 
   1.01125168733520724537227390930342957617`16.715214502436353, 
   0.50397903170532614742484046460518891022`16.412796595638664, 
   0.25218644941599172415338850640333634833`16.112119715496856, 
   0.12765110806476779902375403132833547322`15.816429274326778, 
   0.06373854373488439044845211058441074378`15.514810649441907, 
   0.03177124847068612215892737483470285144`15.212444580742025, 
   0.01609103943699125693843698451516646572`14.916995334591222, 
   0.00796898068116569457728690186112472234`14.611814495819392, 
   0.00400996320949031482918705701546180954`14.313552348507928, 
   0.00200759645372959610853694265941398394`14.0130884922006, 
   0.00099207916519352361217834604507441525`13.706958460092078, 
   0.000496893577396860309396274937267929`13.406675543478288, 
   0.0002500366865477589824580113540212932`13.108415910883043, 
   0.00012600908341409661307573743609998057`12.810814036427887, 
   0.00006180076165614051401544318454625631`12.501406015248381, 
   0.00003100087686004088754910924107567333`12.20178616726318, 
   0.00001552873782486046397616100879560992`11.9015483478134, 
   7.85372503213337061776273285664006`11.60548788261388*^-6, 
   3.92697023760759913048339944645369`11.304469800572024*^-6, 
   1.9423212692126664084520218691928`10.998733256804496*^-6, 
   9.7844200672522155160823153244785`10.700947280639571*^-7, 
   4.9406525212035637310808826475454`10.404196501792573*^-7, 
   2.4794866022544023646085696604583`10.104773956965058*^-7, 
   1.2208234373157780091857007141632`9.797065049280617*^-7, 
   6.118543526641171239633453441681`9.497060244933007*^-8, 
   3.07090554122886199198596992315`9.197678649101729*^-8, 
   1.511198269234051211777595196726`8.889733638548858*^-8, 
   7.6501445423358261743345839206`8.594081831849774*^-9, 
   3.84089759797626518040594336308`8.294844919641282*^-9, 
   1.92731262132595481763150287749`7.995364356492825*^-9, 
   9.4559215157001056971388969823`7.686116049931489*^-10, 
   4.7442041712263994521212415197`7.386575562054708*^-10, 
   2.3952702146631772276087125898`7.0897667050723445*^-10, 
   1.2117033607700341941232516501`6.793808503477295*^-10, 
   5.923885800939487323501670433`6.483018868749416*^-11, 
   2.983605105933178463912939684`6.185153532732284*^-11, 
   1.49720706191717858125263846`5.885694057926333*^-11, 
   7.50421969757109483396075939`5.585717731226464*^-12, 
   3.72961189077170165516759854`5.282075831829615*^-12, 
   1.85129577859781426569175432`4.977888001869259*^-12}}"""

eps_mat = [
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
]

if __name__ == "__main__":
    eps_mat_string = eps_mat_string.replace("{", "[").replace("}", "]").split(",")
    new_str = ""
    for string in eps_mat_string:
        if "[[" in string:
            new_str += "[["
        elif "[" in string:
            new_str += "["
        if "[" in string or "[[" in string:
            _string = string.split("[")[-1]
        elif "]" in string or "]]" in string:
            _string = string.split("]")[0]
        else:
            _string = string
        val, prec_exp = _string.split("`")
        if len(prec_exp.split("*")) == 1:
            prec = prec_exp
            exp = 0
        else:
            prec, exp = prec_exp.split("*")
            exp = exp.strip("^")
        prec = math.floor(float(prec))
        val_float = float(val) * 10 ** (int(exp))
        if "]]" in string:
            new_str += f"{val_float}]]"
        elif "]" in string:
            new_str += f"{val_float},],\n"
        else:
            new_str += f"{val_float},"

    print(new_str)
