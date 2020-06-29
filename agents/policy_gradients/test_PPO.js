import * as tf from '@tensorflow/tfjs-node';
// tf.enableDebugMode ()
import {clipped_surrogate_obj, discounted_rewards, GAE, Buffer} from "./PPO";

describe("PPO Buffer testing", function(){
    let b = new Buffer();
    b.store([
        [10,1,2,4,5],
        [10,1,2,4,5],
        [10,1,2,4,5],
        [10,1,2,4,5],
    ], 0);
    it("adv testing", function() {
    
        expect(b.adv).toBeInstanceOf(tf.Tensor);
      });

    it("ob testing", function (){
        expect(b.ob).toBeInstanceOf(tf.Tensor);

    });
    it("ac testing", function (){
        expect(b.ac).toBeInstanceOf(tf.Tensor);
    });
    it("rtg testing", function (){
        expect(b.rtg).toBeInstanceOf(tf.Tensor);
    });

    console.log("standard functions");
    b.adv.print()
    b.ob.print()
    b.ac.print()
    b.rtg.print()

    let [btchob, ac, adv, rtg] = b.get_batch();
    // btchob.print();
    // ac.print();
    // adv.print();
    // rtg.print();

    });

    describe("GAE function testing", function(){
        let rews = tf.tensor([0.6841638973375666, 0.7492696111949044, 0.8755045802216046, 1.2894141488082822, 1.2962671082961605, 1.062510671048949, 0.7128883151410265, 1.2523650089351577, 1.5280608923072578, 2.5783140936357087, 0.519063123880187]);
        let v = tf.tensor([-0.15255336, -0.24232435, -0.26709902, -0.29354322, -0.25519192, -0.444122, -0.5399537, -0.5923976, -0.62897897, -0.6704067, -0.6708058]);
        let v_last = 0;
        let gamma = 0.99;
        let lam=0.95;
        let expected_return = tf.tensor([9.029394149780273, 8.966058731079102, 8.760120391845703, 8.408426284790039, 7.525900363922119, 6.819906711578369, 6.2177863121032715, 5.902624130249023, 4.9766621589660645, 3.7036948204040527, 1.1898689270019531]);
        let ret = GAE(rews, v, v_last, gamma, lam);
        console.log("OUTPUT:")
        ret.print()
        console.log("EXPECTED:")
        expected_return.print()
        /**
         * PASSED!
         */
    });

    describe("clipped_surrogate_obj function testing", function(){
        let new_p = tf.tensor([-4.339043617248535, -4.540535926818848, -6.73724365234375, -3.94675874710083, -4.163609504699707, -5.531214237213135, -3.526723861694336, -6.439676761627197, -6.322366714477539, -5.0092010498046875, -5.871639251708984, -3.4536795616149902, -5.206577301025391, -3.909111976623535, -3.5758962631225586, -5.319076061248779, -5.777762413024902, -3.23471736907959, -4.789019584655762, -4.054522514343262, -4.650254249572754, -5.627077102661133, -3.5864076614379883, -3.5771889686584473, -6.107110977172852, -5.290377616882324, -3.958043098449707, -4.824031829833984, -4.620998382568359, -4.0216569900512695, -4.153773784637451, -3.8535404205322266, -3.767320156097412, -4.504897117614746, -3.791531562805176, -3.372602939605713, -5.114826202392578, -6.7424116134643555, -8.488542556762695, -4.16473388671875, -4.134071350097656, -4.314373970031738, -7.472906589508057, -7.139802932739258, -9.214052200317383, -5.547598838806152, -7.640525817871094, -5.545079708099365, -4.161508560180664, -3.9219810962677, -3.9062604904174805, -4.013969421386719, -3.394691228866577, -5.292394638061523, -3.768465042114258, -4.423196792602539, -4.3227386474609375, -4.651763916015625, -4.750364303588867, -4.34284782409668, -4.075653076171875, -3.837770700454712, -3.815253257751465, -5.6972222328186035, -6.262546539306641, -6.538112640380859, -4.680283546447754, -6.0419769287109375, -3.685710906982422, -4.074259281158447, -5.701682090759277, -5.125494956970215, -5.026112079620361, -5.616322040557861, -6.701656818389893, -4.650474548339844, -4.002551078796387, -4.097878932952881, -3.630049705505371, -2.9034082889556885, -2.834946870803833, -4.501170635223389, -5.428349494934082, -5.780641555786133, -4.888271808624268, -5.567448616027832, -3.171645164489746, -7.549286842346191, -4.379704475402832, -4.058135986328125, -3.9617886543273926, -4.383366107940674, -5.07137393951416, -5.997598648071289, -5.026457786560059, -3.6595845222473145, -3.473135471343994, -4.67906379699707, -2.9033889770507812, -3.164297580718994, -7.732839584350586, -4.871603012084961, -3.559910774230957, -5.773124694824219, -5.147886276245117, -4.715953350067139, -4.249462127685547, -5.527668476104736, -4.570603847503662, -3.529876470565796, -4.075125694274902, -3.3746986389160156, -4.054471969604492, -6.875231742858887, -6.320979118347168, -5.109482288360596, -4.150021553039551, -5.137249946594238, -4.568498611450195, -3.4161205291748047, -6.616983890533447, -4.902645111083984, -3.56472110748291, -4.4878387451171875, -8.59241771697998, -4.492022514343262, -4.808323383331299, -4.4817376136779785, -4.504720211029053, -4.238691329956055, -4.915308952331543, -6.35564661026001, -4.966209411621094, -4.9823994636535645, -2.7951724529266357, -4.381099224090576, -5.2546820640563965, -3.804327964782715, -5.112278938293457, -3.2638843059539795, -4.491507530212402, -4.694483280181885, -4.931539535522461, -4.419741630554199, -3.2343339920043945, -4.649767875671387, -5.192127704620361, -5.265573501586914, -6.170458793640137, -5.303611755371094, -3.847499132156372, -7.108244895935059, -4.692089557647705, -6.18914794921875, -4.021666049957275, -4.225654125213623, -6.104302406311035, -6.519650459289551, -5.315721035003662, -6.035949230194092, -5.0077290534973145, -3.969741106033325, -5.780216217041016, -5.6011762619018555, -6.103880882263184, -5.235359191894531, -5.407719135284424, -5.047948360443115, -5.919814586639404, -5.374866962432861, -3.9279487133026123, -5.5806169509887695, -4.2334370613098145, -4.784089088439941, -3.4967801570892334, -5.787718772888184, -4.3389177322387695, -4.1212158203125, -3.157172679901123, -4.126549243927002, -6.0074262619018555, -5.283196449279785, -4.125062942504883, -6.757754802703857, -5.957632064819336, -4.672752380371094, -4.9071760177612305, -6.077734470367432, -4.449507236480713, -3.8466804027557373, -4.363615989685059, -3.514024257659912, -3.8917598724365234, -3.8104262351989746, -4.196917533874512, -4.190946578979492, -5.807584762573242, -5.515296459197998, -3.432791233062744, -4.086897373199463, -3.8364357948303223, -3.9486048221588135, -3.972853183746338, -4.774966239929199, -4.298800945281982, -5.295511722564697, -4.7293009757995605, -4.363648414611816, -5.74339485168457, -4.394820213317871, -6.372306823730469, -3.052272319793701, -3.7940802574157715, -6.14300012588501, -4.895503044128418, -5.245083808898926, -5.379555702209473, -4.07666540145874, -5.634641647338867, -4.92253303527832, -3.634019613265991, -4.827127456665039, -5.303171157836914, -5.697277069091797, -3.6115286350250244, -2.8344781398773193, -4.99655818939209, -3.9920239448547363, -3.9852066040039062, -4.590890407562256, -6.454550743103027, -4.536507606506348, -5.226573467254639, -5.07235050201416, -5.098360061645508, -4.3683600425720215, -3.6871654987335205, -4.979193210601807, -5.7461442947387695, -5.315041542053223, -4.916378021240234, -5.912899017333984, -4.259067535400391, -6.128222465515137, -5.187898635864258, -3.0649213790893555, -3.5657715797424316, -6.05576753616333, -4.907651901245117, -5.4793291091918945, -3.148923635482788, -4.685833930969238, -3.8017845153808594, -3.7202272415161133, -4.793081283569336, -4.375473976135254]);
        let old_p = tf.tensor([-4.339043140411377, -4.540535926818848, -6.73724365234375, -3.9467592239379883, -4.163609504699707, -5.531214237213135, -3.526724338531494, -6.4396772384643555, -6.322367191314697, -5.0092010498046875, -5.871639251708984, -3.4536795616149902, -5.206577301025391, -3.909111499786377, -3.5758962631225586, -5.319075584411621, -5.777762413024902, -3.234717607498169, -4.78902006149292, -4.0545220375061035, -4.650254249572754, -5.627077102661133, -3.58640718460083, -3.5771889686584473, -6.107110977172852, -5.290377616882324, -3.958043336868286, -4.824030876159668, -4.620998859405518, -4.021657466888428, -4.153773784637451, -3.8535404205322266, -3.767320156097412, -4.504897594451904, -3.791531562805176, -3.372602939605713, -5.114826202392578, -6.742412090301514, -8.488541603088379, -4.16473388671875, -4.134071350097656, -4.314373970031738, -7.472906589508057, -7.139802932739258, -9.2140531539917, -5.547598361968994, -7.640524864196777, -5.545079231262207, -4.161509037017822, -3.9219813346862793, -3.9062600135803223, -4.013969421386719, -3.394690990447998, -5.292395114898682, -3.768465042114258, -4.423196792602539, -4.3227386474609375, -4.651763916015625, -4.750364303588867, -4.34284782409668, -4.075653076171875, -3.837770700454712, -3.815253257751465, -5.697222709655762, -6.262547016143799, -6.538111686706543, -4.680283546447754, -6.041975498199463, -3.685711145401001, -4.074260234832764, -5.701682090759277, -5.125494956970215, -5.0261125564575195, -5.616322040557861, -6.701657295227051, -4.650474548339844, -4.002551078796387, -4.097879409790039, -3.63004994392395, -2.9034080505371094, -2.834946870803833, -4.501171112060547, -5.428349494934082, -5.780641555786133, -4.888271808624268, -5.567448616027832, -3.171645402908325, -7.549286842346191, -4.379704475402832, -4.058135986328125, -3.9617886543273926, -4.383365631103516, -5.071374416351318, -5.997597694396973, -5.026458740234375, -3.6595842838287354, -3.473135471343994, -4.6790642738342285, -2.9033889770507812, -3.164297580718994, -7.7328386306762695, -4.871603488922119, -3.559911012649536, -5.773124694824219, -5.147886276245117, -4.715953826904297, -4.249462127685547, -5.527668476104736, -4.57060432434082, -3.529876470565796, -4.075125217437744, -3.374699115753174, -4.054471969604492, -6.875231742858887, -6.320979118347168, -5.1094818115234375, -4.150021553039551, -5.137249946594238, -4.568498611450195, -3.4161205291748047, -6.616983413696289, -4.902645111083984, -3.56472110748291, -4.4878387451171875, -8.59241771697998, -4.49202299118042, -4.808323860168457, -4.4817376136779785, -4.504720211029053, -4.238691329956055, -4.915308952331543, -6.355646133422852, -4.966209411621094, -4.982399940490723, -2.795172691345215, -4.381099700927734, -5.254682540893555, -3.804327964782715, -5.112278938293457, -3.2638843059539795, -4.4915080070495605, -4.694483280181885, -4.931539535522461, -4.419741630554199, -3.2343337535858154, -4.649767875671387, -5.1921281814575195, -5.265573501586914, -6.17045783996582, -5.303612232208252, -3.847498893737793, -7.108243942260742, -4.692089080810547, -6.189148902893066, -4.021666049957275, -4.225654125213623, -6.104302406311035, -6.519650459289551, -5.31572151184082, -6.035948753356934, -5.007728576660156, -3.969740867614746, -5.780216217041016, -5.6011762619018555, -6.103880882263184, -5.235359191894531, -5.407719135284424, -5.047948360443115, -5.919814586639404, -5.374866962432861, -3.927948474884033, -5.5806169509887695, -4.233437538146973, -4.784089088439941, -3.4967799186706543, -5.787718296051025, -4.3389177322387695, -4.121216297149658, -3.157172679901123, -4.12654972076416, -6.0074262619018555, -5.283196449279785, -4.125062942504883, -6.757754325866699, -5.957632541656494, -4.672752380371094, -4.9071760177612305, -6.077733993530273, -4.449507236480713, -3.8466804027557373, -4.363615989685059, -3.514024257659912, -3.8917598724365234, -3.8104262351989746, -4.196917533874512, -4.190946578979492, -5.807584762573242, -5.515296936035156, -3.432791233062744, -4.086897850036621, -3.836435317993164, -3.9486052989959717, -3.972853660583496, -4.774966239929199, -4.298800468444824, -5.295511722564697, -4.729301452636719, -4.363648891448975, -5.74339485168457, -4.394819736480713, -6.372306823730469, -3.0522725582122803, -3.7940802574157715, -6.14300012588501, -4.895503044128418, -5.245083808898926, -5.379555702209473, -4.076664924621582, -5.634641647338867, -4.92253303527832, -3.634019613265991, -4.827127456665039, -5.3031721115112305, -5.697277069091797, -3.6115288734436035, -2.8344779014587402, -4.996558666229248, -3.9920237064361572, -3.9852070808410645, -4.590890407562256, -6.454549789428711, -4.5365071296691895, -5.226573944091797, -5.07235050201416, -5.09835958480835, -4.3683600425720215, -3.6871652603149414, -4.979193687438965, -5.7461442947387695, -5.3150410652160645, -4.916378021240234, -5.912899494171143, -4.259067058563232, -6.128222465515137, -5.187898635864258, -3.0649213790893555, -3.5657715797424316, -6.0557684898376465, -4.907652378082275, -5.479328155517578, -3.148923397064209, -4.685833930969238, -3.801784038543701, -3.7202272415161133, -4.7930803298950195, -4.375473499298096]);
        let adv = tf.tensor([1.0065109729766846, 0.9859451055526733, 0.027030184864997864, 0.8050152063369751, 0.8838555812835693, 1.3369035720825195, -0.9304555058479309, -0.5346431732177734, 0.5667810440063477, 0.15791961550712585, 0.9194470047950745, 0.7536730170249939, -0.9341413378715515, 1.045669436454773, 1.6203103065490723, 1.0015923976898193, 1.2671029567718506, -1.0484495162963867, 2.8040688037872314, 0.1753043532371521, 0.9818244576454163, 2.082509756088257, -1.3527436256408691, -0.19421818852424622, -0.8871544003486633, 0.09353470802307129, -1.2605526447296143, 0.7968606948852539, 0.9734120965003967, 0.23262012004852295, 1.9641433954238892, 1.2541946172714233, 0.8072471022605896, 0.7606706023216248, 0.09645657241344452, -1.09466552734375, 0.6346076130867004, 0.23327629268169403, -0.002773603657260537, 0.21860140562057495, 0.28260067105293274, -0.0769302099943161, -0.3176476061344147, 0.24986375868320465, -0.7371020913124084, 0.40454232692718506, -1.2568172216415405, 0.7259033918380737, 0.7238482236862183, 0.27911442518234253, -0.6609286665916443, 0.33820807933807373, 0.5722842216491699, 0.1328728049993515, -0.23385308682918549, -0.5437725186347961, 0.9018366932868958, 1.0864099264144897, -0.5852068066596985, -0.31680116057395935, 0.6115714907646179, -1.0383867025375366, -1.5596201419830322, 1.13255774974823, -1.085576057434082, -0.2151438444852829, 1.4949917793273926, -0.5271949172019958, 1.206826090812683, -1.0081521272659302, 0.05030704662203789, 0.20718702673912048, 0.056077420711517334, 0.6876017451286316, -0.8538810610771179, -1.0531420707702637, -1.4089362621307373, -0.3708699643611908, -0.7551955580711365, 0.7272623777389526, -0.9626947641372681, 0.5953661799430847, -2.0086312294006348, -1.2614527940750122, 0.368596613407135, 0.33869218826293945, -0.9202731251716614, 0.1613682210445404, 0.8258294463157654, -1.5101501941680908, 1.0392751693725586, -0.9083332419395447, 0.26724106073379517, 0.8568511009216309, -0.30793362855911255, -1.4958648681640625, 0.40804967284202576, 1.0734909772872925, -0.7696784734725952, -0.9486907720565796, 0.9213021993637085, -1.0876708030700684, 0.4575115740299225, 0.02488458901643753, 0.08835636079311371, 0.6556825637817383, -0.3902539610862732, 0.06415163725614548, 0.586112380027771, -1.2349148988723755, 1.025376558303833, -1.3212721347808838, -0.8507990837097168, -1.606468915939331, 0.057237409055233, -1.3887619972229004, -0.264814555644989, -0.44110170006752014, 0.5242993235588074, 1.0466084480285645, 0.9966705441474915, 0.9259465336799622, -0.26031237840652466, 0.9489552974700928, 0.039854466915130615, 0.5243566632270813, -2.0204086303710938, 0.23858368396759033, 2.623870611190796, -0.038085803389549255, 0.305417001247406, 0.3610512912273407, -0.47961410880088806, 1.034309983253479, -1.6669553518295288, -0.8407503962516785, 0.6448923945426941, -1.9323092699050903, 0.1491318643093109, 0.7410401105880737, -1.5825097560882568, -0.8861017823219299, 0.6657591462135315, 0.8797356486320496, 1.0179795026779175, -1.9285119771957397, 0.5814465880393982, -0.8201520442962646, -0.48842108249664307, -0.6914119124412537, 0.6406974792480469, -0.5204097628593445, -1.7887992858886719, 1.0465569496154785, -1.057499647140503, 1.119848608970642, -1.1056865453720093, 0.7413113117218018, 1.3446835279464722, -2.1389877796173096, 1.0224847793579102, 0.6841931939125061, -1.8846596479415894, 0.45046329498291016, 1.2339938879013062, -0.44926732778549194, 0.8211067914962769, 0.9557615518569946, 1.1522645950317383, 0.014936591498553753, 0.026089133694767952, 0.7269102931022644, 0.8357559442520142, 0.7717519402503967, -0.2750268280506134, -1.9865285158157349, 0.17144130170345306, 0.1092931479215622, 0.824110209941864, 0.8583570122718811, -0.5962094068527222, 2.1278774738311768, 0.7256842255592346, 0.9547094106674194, 0.03193120285868645, 0.2272854447364807, 0.9558219909667969, 0.20514284074306488, -1.0332690477371216, -2.114875555038452, 0.4282608926296234, 0.2578493654727936, 0.3910827338695526, -0.7001000046730042, 0.3305019736289978, -0.5862019062042236, 0.091060571372509, 0.756587028503418, -1.754971981048584, 1.0874292850494385, -1.3223776817321777, 0.9728848934173584, 1.4183069467544556, 0.37126868963241577, 0.6720842719078064, 0.9688753485679626, 0.07897017896175385, 0.7918509840965271, -0.5246877670288086, 0.7628635168075562, -0.8290926814079285, -1.7438640594482422, -1.9103139638900757, 0.34815749526023865, -1.766645908355713, -0.07902766764163971, 0.5739012956619263, 0.6393707394599915, -0.31471267342567444, -0.5532195568084717, 0.8837961554527283, -0.6160646080970764, 0.7715281248092651, -0.5749982595443726, -1.6739885807037354, 1.3889858722686768, 1.0758445262908936, 1.0290793180465698, 0.7397835850715637, -1.0857372283935547, 0.38151678442955017, 0.70003741979599, 0.6613463163375854, 0.876525342464447, 0.6031709313392639, 0.9084427356719971, 0.7473898530006409, 0.8146297931671143, 0.8534336090087891, -0.23100444674491882, -0.2375384271144867, -1.3783522844314575, 0.6263512372970581, 0.7375660538673401, 0.9098769426345825, 0.8028350472450256, -2.119250535964966, 0.8715000152587891, 1.296677827835083, 0.6512280106544495, -0.9149819612503052, 0.23733510076999664, -0.9640430808067322, 1.8351036310195923, 0.39241939783096313, 1.3394556045532227]);
        let eps = tf.scalar(0.15);
        let expected_return = -0.09097351;
        let ret = clipped_surrogate_obj(new_p, old_p, adv, eps);

        console.log("OUTPUT:")
        console.log(ret)
        console.log("EXPECTED:")
        console.log(expected_return)

        /**
         * PASSED!
         */
    });

    // describe("discounted_rewards function testing", function(){
    //     let rews = tf.tensor([0.7063738555531017, 0.7146117017284268, 0.5721372607687953, 0.8387586085125804, 0.6346586222120094, 2.1229582789121193, 2.441930156690068, 1.6537365323340054, 1.5682498665337334, 1.2107448143811779, 1.343330289462756, 1.13445747345977, 1.0818317277735332, 0.18282593561889365, 0.5618275294240448, 0.34446033439890017, -0.808419511723332, -1.2188085476533161, -0.9603651840458042, -0.4430050302413292, 0.16397659063659376, -0.27013589980488173, -0.72087515474268, -0.24246899817080703, 0.6902697704121237, 1.8101793446490773, 1.4559603539892123, 1.2562842830229783, 1.2153714538915665, 1.0748328937406768, 1.3603848784754518, 1.446027693474025, 1.0123445062170502, 0.9979044471954693, 0.7116143221908715, 0.30195935124793327, -0.9187922266079113, -0.07207446905958934, -0.7989612227000179, 0.6443257182472735, 1.2275344460751512, 1.1297683263357612, 1.0802633118262748, 1.063619704679877, 0.6158940645094845, -0.1331166300136829, -0.17656569720420523, 0.6562242807180155, 0.5205088874194189, 0.9896286344097461, 1.780463573700399, 1.553874729340896, 1.8017109602267736, 1.8481227439377108, 1.4558905444297126, -0.7353320113470545]);
    //     let last_sv = 0;
    //     let gamma = 0.99;
    //     let dr = discounted_rewards(rews, last_sv, gamma);

    //     let expected_result = tf.tensor([30.477680206298828, 30.0720272064209, 29.653955459594727, 29.375574111938477, 28.82506561279297, 28.47515869140625, 26.618383407592773, 24.4206600189209, 22.99689292907715, 21.64509391784668, 20.640756607055664, 19.49234962463379, 18.543325424194336, 17.63787269592285, 17.63136100769043, 17.241952896118164, 17.068174362182617, 18.057165145874023, 19.470680236816406, 20.637418746948242, 21.293357849121094, 21.342809677124023, 21.83125877380371, 22.77993392944336, 23.254953384399414, 22.79261016845703, 21.194374084472656, 19.93779182434082, 18.870208740234375, 17.83316993713379, 16.9276123046875, 15.724472045898438, 14.4226713180542, 13.545784950256348, 12.674626350402832, 12.083850860595703, 11.900900840759277, 12.949185371398926, 13.152788162231445, 14.092676162719727, 13.584192276000977, 12.481472969055176, 11.466368675231934, 10.491015434265137, 9.522622108459473, 8.996694564819336, 9.222031593322754, 9.493532180786133, 8.926573753356934, 8.490974426269531, 7.577117443084717, 5.85520601272583, 4.344779014587402, 2.5687553882598877, 0.7279118299484253, -0.7353320121765137]);

    //     /**
    //      * PASSED!
    //      */
    //     // it("Result matching test", function (){
    //     //     expect(dr.dataSync()).toEqual(expected_result.dataSync());
    //     // });
    // })