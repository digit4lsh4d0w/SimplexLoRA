# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% libroFormatter="formatter-string"
# %env CUDA_VISIBLE_DEVICES=0
# %env TOKENIZERS_PARALLELISM=false

# %% [markdown] id="95b4cfd741795038" libroFormatter="formatter-string"
# ## Initialize PolyModel

# %% id="1a5c7a99-5208-4d22-ac15-bacebe1b52f9" libroFormatter="formatter-string"
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, concatenate_datasets
from peft import PolyConfig, get_peft_model, TaskType, PeftModel, PeftConfig

model_name_or_path = "google/flan-t5-xl"

r = 8  # rank of lora in poly
n_tasks = 4  # number of tasks
n_skills = 2  # number of skills (loras)
n_splits = 4  # number of heads

batch_size = 8
lr = 5e-5
num_epochs = 8

# %% colab={"base_uri": "https://localhost:8080/", "height": 241, "referenced_widgets": ["dc5d4672fcd149239cfe1a837094ce53", "eded01d7629e4a4faad592e8e20a3ca3", "5d1e94d40f514faaa5819096f167d29c", "f98f73664a974ae7804e494425fbe20d", "1c0bd751a3294b8ea0cf828866169121", "6c3ed2de06fe40c09315ff72d43d5c8c", "2e3d6b5d46db4295829002fc311a9c74", "5ff0d4da7342457089f0961b189307f4", "a08c4e6628bd440fb31eebbb2693f327", "379357ab63f5479fad469c181b054bb0", "f860e1c3467348f0802b733fbef45c15", "567e165c27a4494bbf4810ecb7de40cf", "015fd47fdbdf47c5a619eff218052b45", "ec33a4325b6f4dcfb8a9fa4c80a5c704", "3241189c875a471ab0831f0f4411d2d3", "268fe971a0bc45c6b7c37586e0f9da49", "8851d4a04cb9410c849b6606a812c52b", "3b5ab7d9f27944d8ae1b172231c9c6fc", "85f57b44dbe442a4952c65e1db4c1176", "3f173a7293cd4ff8a54da8c8174cfb43", "40ac1e38c100435fbe95b669c69a31c5", "a634013728be457ba590aa333908addd", "376242d1cfd74c88aaeaa76a6813d855", "a17443b5713d4b60aeb85da3adce6cf2", "91f821fb888046b6a2f8ade2cc58db2d", "4bea407148e846babefdc88eff8a9131", "26a75e6f6628472b91f3214505afa935", "c7c0b0fd45dc448eb9456f58e36fb3bb", "a14f26db56d04b8b840a9ce366e913e6", "09fa4b156f174dbcacdf976f2b39a280", "c9ca89486def4220967599e5b159b980", "558b98eb76654045a5eae24170a5dc9c", "7edf2ac4dd264843a7838a0130668757", "c3aa97f46a60409091dc4d33a946c6d3", "e6315c5d217b4922b461c9ac22528e62", "04c34c92e4374c50bd0636c72953a8ba", "32a6ac79c27e47c1a4b32098bfe25807", "bb99715a25d94422b0048de94f2fe563", "637bbd213f3345178742523d055993e6", "6113f1920c5743aa8f2c6cc9739029e1", "24bbd7b810b34c4c9baeed628961c64b", "1c63e99470824a3aa0f98a94862733d5", "98c677014f1a48ac804cec0714a22172", "a4097270b9b947b0ad0b3b5d217eecc0", "8eaed8cbbf1943328dc80fc43bd5b97c", "6b1972a032af41de9bf99a6582c53f39", "b4eb16a8153048ea9aa5c9d43b44820c", "9bd66a63faf9416d9e774a5d8221c5f5", "40b859a2fd68457db691bb5e7eb23591", "533151b377d64d3484772b3173dab306", "2cd302d306e3440dac4b70fc46741544", "41b36d52e98249b1b506d369d2d8e994", "ac83130fdd374b7c8f41e0f8f011ecae", "66b0e949143e46faab77458a49a9fe1a", "abd34fa3e94c49869ea7cf514dba6d1d", "8ffe87ece7e54294a160540fbbbe124b", "9c3c68da285449958a3d8745bbc50305", "ae517eef5a004b16b4ae34cdf2aa851e", "08a572aefb63488d8125ae3b881c0729", "f2131e286f704514a61b5af0785dde8b", "43d9b3de4a6949f787d9733d1ae4d18e", "06716294f2244cc48f78af918cc063f2", "e69b0005e91a478297d17e4089cda650", "f92f9afc2c694f0cbcdf4ebcca98221e", "7ffe5fd0a64c40cebc784eca83154069", "d622b006621e4110a157fb4cb43c9762", "874e05e0b861466ba57a08d8f5a5b7ee", "8ebe69a07de64c3cb6dfd6433e222186", "2aceeebfd0dc42fcbbc1b3a7e1f54c56", "93c6f7c0d1ba49a295ae60a73bf509a9", "6c3ebb812cfd493bb954a6b1d7455c72", "a27edbdb4c824979b1b56e8fbd867595", "5bdf79c178074ebf8757936190bc37b3", "3c076081fd7942e184f8d4f171a17e1c", "0a03bee83ddf4ad297bfdc9b4de3b075", "6f59ae0a20cf4cc5859925e3259291a7", "49ac9897f49843fd8c5fed4bcdfdbb56"]} id="89a1d2c6-0d35-4254-b9fb-035a426d86ae" libroFormatter="formatter-string" outputId="fc90c2cc-9cab-40ed-bf4a-d76bec85b72f"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="29d701a4-7a4f-4eae-84bd-9e3a02b7ffca" libroFormatter="formatter-string" outputId="63898f68-926e-40c4-ca13-ffd1df32fcce"
peft_config = PolyConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    poly_type="poly",
    r=r,
    n_tasks=n_tasks,
    n_skills=n_skills,
    n_splits=n_splits,
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# %% [markdown] id="aa695c2d-cf9c-432c-ab74-7e89f816ba13" libroFormatter="formatter-string"
# ## Prepare datasets
#
# For this example, we selected four `SuperGLUE` benchmark datasets: `boolq`, `multirc`, `rte`, and `wic`, each with a training set of 1,000 examples and an evaluation set of 100 examples.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["d6250bff76d7454a8216572ab28e4a72", "384d10ea2a354f24bae33c3a1d564b82", "a2deecc9aa3d42d381d78199f6e29d1c", "17fc618034bf4aadaef811b0e7c80eed", "6757bc0834fc4e69b7b588ae6de14ec9", "b9ec517b4b084d548525ac41381ef69e", "6f3679fe9b60498da864bda9ba6d899e", "ef16f8bac38044c3b6a092caf5da320b", "19e50ecdda3b493184611d97724ac1fc", "c2ed87d5599a467bba084cddb9e40713", "2cfc492ab0ed454dbf2c4da18cd24d02", "91e6e0685a4c4d26b6154d3ed18418eb", "ba1864322c0d49fd915e9dcc2469ef6f", "67b108def57749edb2564b3e507959a3", "c04a17f40c974f378c60858473f49fd0", "4189c6e9c59e44d3a776b49c38cc8f06", "7550307b4e894844b8d032df7eea6d82", "bcf20733fb504a71be5cf0455928b587", "cc6c1b2d4fcb4ffea016a139738e1ead", "6bd3da08b5074e81bffbfe6d92b8ce8b", "03d340641a414362b0356e8178148d9a", "61fec3b2596c4803924ed1fb087d52d1", "4fc54c5844aa44f2b335824c3544a334", "736443c7e26642379ca66ed3e5dd34cb", "3c3d747638004a08a898cff7c6f59acc", "1cb2dcb242334f46b7f195929dd1f341", "f577bbac4eab439b9ccea0a49eb99d86", "8822d3a8fa794fc0addb5885a862d205", "10436da727ec45c8a5e8b783696636d5", "0622e1de75f34da590f241232613cf5e", "5b868029728541dc9da977312da38cf0", "33add0384c36462ab44fe3e0b03f63c7", "e227fb95b00b4af8b82286c75db84611", "54d4afa42e9346578c0a1a193ee8caea", "2918f1fd9e104c09967d698e11728785", "170853712c0c4a8d997696f74090d7c6", "fcdb61acbf0d470f881ba8f283360e0f", "4803c9aece3346488295338254217aff", "d725549ca34a4d54ae684e7e4741be29", "d94392afd01246ff942af838a995379e", "7eaee1cd25d0442092846922cdd6c413", "483b9c219aa94ea1952e3534a02395aa", "10d7a41588744be1b29678b4a9dfdd27", "c4fce7e5a2b44835ab8723e0022d1e50", "ba3a7258734b4edb86b8eef074d65222", "4b3dc87d00ed42b0956d0bfa39bd466f", "2c5eaf38e66d492d8661852cacc4e527", "b7f169f931074d1283cbfe912f11ba98", "e86771ea303a4b1b86ecf5128f3ea421", "b3a18689eefa4660997034094df0df04", "91cb4404dd794d22b2bbaf31eee207b5", "d8594695c03e4fb7965dcbe04074d4eb", "f972a2e10dde405c8aec8f7cd1be4317", "6c9a5a39cb4841ba8a0b93283be0cac2", "2deb046362ed4570a3f550f4f288529e", "2cdddd0398e045a6a13124bf6fd85506", "a1f5618e59d148409d6ccf4bfffca2fa", "c870763add1745d9acfc2762f468c984", "be70aecc5b294c4c93b0dcc09d6d1cb3", "37b18b2ae9504b6c91066798a19a1319", "b17eef10573f44689ce6add6231eaa19", "3e8f08e000f248b59331d2430bbc8e3c", "d8886d5af17f4468a26554831c9c05f1", "b1cad4191755493893bbb46dcf27e03b", "589120da6f464686bbeff0d44643d17d", "ea89ea173ff3482c8a9c91dfb15946b2", "b38693117df44071b7baaf123215ea60", "9d716c9e43e04a6c9496620633ca28be", "4de322f2413f44bcb03d41dcc8ff1963", "3087335b98964b9eb4da474487ca4864", "c010fad90578489cbfeb0764e3a11286", "627f32f04b544b4db834b79645f36733", "7d39222af2474a68b0db99f407ccf380", "775687d3962242d6aff3feb0627754a9", "92bb9917ab194a1eb1dc6fc5c4c4195d", "43717a0ae4c043f9947d8fd844d71997", "3f0008931054433c838a6633ca1347e6", "da0dfe11648d4ae8a70852ce1fac87b0", "29ad37e81b7f44a9aafa982b52f05a7a", "bb468a6fe3f04692a211d5519aec455a", "b1ad08dbe61b4064985ecdaa119870d8", "d91093e80b814a018edaabe49f529ef5", "1e3f013edc6341a0837af33ff4866d0b", "1ed6f4595ec540729d776a81db96c403", "cd7371ff8504454292559c18adb76645", "d4994c6d7fa240d0ac6bb31f5c835192", "ee1a9269b6c843e28cc49f3b5f17da96", "a12a6638b2a54e88a020be42c139646e", "2f2dc993705b447aa771cf0cc13c3b1d", "69dffb139cab46b1b93bef960f702655", "e7dc30a09a64401393e43618b51059de", "75c397c506a04d0d9ca62e8d7f990813", "eb7e509acfea4e1bb2f59b2fde11603d", "b75a3d92aaa64a3098c6e1aabbc50856", "3b056671c3fd41aeb4d6da821d562b95", "9ac87f9e5b7847aaa90e3208ad405c23", "cb21cadacb294854b304af8df2157299", "30b3667258174beaa01322ffa055759b", "48cb432d95dc49deae6077fb5c76bec3", "763eec2b33234fd5ac192f25489b2844", "f7fde1c95e3946659c6208fc52c254e9", "6e1cdd75ae2246278c80f8e5d4e340b7", "6af159109303490eab7192815fce0d6b", "315575253cb9433d81f7d26770907f29", "daf61ecd65bd41d9829e8a1872b82f33", "54823295494d441fb9f26a70fb2c3973", "df13aabfde0140d68db8b5a69759091c", "cadbbc94bfd24daa933bb7d188dcdf92", "f271a7dc607b4c05b02f6c5621203bd6", "4e60ffab53dd4aba9e094098ed5297e6", "a91dd2c07c51483eb326d010a82e2920", "4188c097387a408dae680f67bd97752b", "2a6cf3f2b4c349a9adc26351c2f0b222", "8efe1cb67f30446a8fdaaf96782e843d", "bdecbe50f693451b87bd331fd9e684ba", "ff00f4c2c63a467098b119ae2259f529", "d6754db1364144e69e3ab320aa1faeb9", "9f1feddbe0d449a0b93fc5b1027e4319", "1507ea5f89534b10856b99488ed5da65", "f69205c589cf42829eea248f378a1436", "85843fd6df264d25a4642cbeee260459", "c2a415c28fe0418bb6032d4b91efbb07", "b02796bf6e5547cf9418d109ad772537", "de7a527901014a6b8abf6b714bd09535", "143eaab6e65f4c8eb7a4314cadf323ba", "e0603f00ceb2468692a36abd7bafaba8", "8abf122ed835496aa09945ba8edd4688", "663b9bd2f1af4df4b757624f53c2f2b8", "c352adea84b043db8d43eb1c36d4bd4f", "8fda102684834d21b4efbb472823ced9", "ee7eccaca57b4460becfe0d5d5afb3f9", "554a1ef44aea4ff484f0944878bb58a2", "8ae15293ec5e4296a625087e7d965249", "db067e50373840b19d6925deb950a20a", "ab409caa3be24c18becaf9146b1ae69c", "63236709413940f59d2622f2927c8d55", "3e6d67d246e54fa497f4398e3aeddb00", "118967bdd4a348858cc7572d36c1b736", "8cc3c0558720400e9ed89170883f6370", "59ee159eec154095a368efceb9d1e042", "f78c80dd733d48a687d8a47bfc792ea4", "9da9d7eb9a97495194a3ac4a2786e1de", "3f1d345037604e01911ef344f3b51742", "eeff19b29a6a4c6d9ae34e365c78c310", "a098c442cc8149a5aa562c86fc64528e", "badae7117ab644ffa80d099c17397329", "13ccc23d506248d5b66ffe7732ead149", "16938b11881741af8e6633094a4402dc", "8d796692add94119a4e9fdc6530a6878", "9202902fd37446cda1678c0d83e0c641", "2cd168e5e3c4481ba151aa8a655e7ce8", "f773b949a9cf46ca9fd56476398a3191", "c6a00e1b00684bb7930fb27d6499932e", "4395227decf642f7b8fbc6616f9ec826", "205a7844670249bf83d468fe3af0e139", "28a78fb894a4413e960c1e40d7df8173", "46bd38ac919e4e66a72226c1f0da67d6", "9557a05279544cd8a5f2ba4d3429f576", "2e0e7f437b5d4e4086b38ee6da51dc4f", "73a0ea365433404babb83a2d1caa9c66", "855a155046904aaa9b91b01dd6a86088", "d49e405525ec467bb7a69a9aaedf82d9", "b8ccdebb7b11490e8222ff79ecfc9a33", "9edc1d5792644f79bc04d853b13dac46", "1b78ff7e32254777abe2c802b6879b9c", "32168837680e41eaad4e5e4cdf09877d", "319bf5c6332f41958974d9c3af87a382", "c1e065fa36344f509e7863c3ec0428b8", "5b9510c694a24afbb1c8318fea1a1bc5", "f12d118a6b3f46b580fbe2018f4cf5e9", "726e82f2b6c94e8eb5620c18872aabb6", "12bb708857e84e5b893ca3e9ff176082", "53acddb088564a73aef61618797bfe85", "fb208f3792174ac2bdfb077450b2218f", "0fa8ddd8da924c089221611c98e7da6e", "b4a5fdee693d455686507306da804b17", "045c5ac6981440c996ac7dda054fc112", "f6a6f03ea7f140189a277742ff7082f8", "ecff5b6dfb784c96b69d1a39b7acb171", "43a32f261eaf42af978a6bf98502b1fc", "8b7bb3fb502f4c3185709d6c40638d70", "bf1d0b17974049c6ac5653ac18f1169c", "32672cb3395045bcb9c2d370032356cc", "4728eddf9fcc49d68d71a30379f08335", "8676e2232dc242c39d4f19b0eea90dff", "167e67c018e441d8baab4127b25773c6", "88405a75b92743e589f424ee8c4d4d79", "b5e3536d816c45488bb83336eaa5d53f", "b9bed2c861df4c019ab4fff46b11a1a3", "35bf380a7c6243859a459560288ffe49", "b65af12dbf9143778412adb7b4c0bfdd", "cb2433a0096845468b26c4bbde625ed8", "8d4df1cf62d2427b8e850b031164ff97", "0fd8ca256b8b49e1906a2a8e21156164", "9801a82eda354815b6b3abbc8d1e0140", "b10768b6bc654da8b822b4878889639b", "c56ea889f51848e4aed83bcc46c83395", "c75508086f6f4406a0aa9ce5a391e0ee", "d92dbe59ace74f598efc7fbedb4c5ee6", "675c937cf0ef4bf582f4bf90df6fa28e", "e1cf760846bd4ba988c29665a6593220", "768735ea9663429ba9f24efd86682f71", "4d2a10e9307a47f4a1cbf512380c65bc", "e8738d4181b04545a0418c1dd5b6b1b5", "2aab7297963040f1900f0bc1f24e7b2a", "8182a23b5be640cc8a48c09a4ed9585c", "c3e198ae77684d61bf5fc30a35d8fc11", "aaa71d52156549a4b8d7aad390497ac3", "6387cbc474144e59aac5e3b42e714887", "efc4f9ade28a4bb2a67c0ae4ceecbf28", "7c105cfe1f344bf7896c7ddc0fcdc322", "c1f71dcbe98f4ee9847af6b800979e06", "e12dbfa20e9d40448366c9528c1a2c02", "b3d7aad60442432e96c8c8bd3ead8427", "0d40464e81fe4c06ac3400204116f243", "69a0d832b77e47b8a2adcf47efe3f7ab", "947f13ea22654b4ca6fca7ebed29d64e", "45becb2c72714dfcb721b3a20a92d28f", "4ddb5f1d8260448981c67308bcedecbf", "d43db93804c24908bb6d75f26b640199", "71d656fe70004c6db5d23d86bc6b108b", "9f2ecadf1f3f4e399aad1882f2fe9b00", "8826a0177e334508932a43563d2ae97d", "f9ecbb00f95548d5b0c5cad345b1e38a", "ac78d9726bc541df9907425442d3a51a", "854937001e534695b08ee25f6e443962", "06b21316e1ef41c9b7c9d943a9ff91ec", "d38618d4c0e64b7baa61c0eca47427e5", "86e235a532f347c781d6654c3ac25ba3", "1727de01c47144b2958babfb91e887cb", "5ac310c605f64948ad744bc1f196441d", "32c1cc0d5327462d9175c74b91d67c4d", "910d8a34abaa4f92a899dd4f5ab03d74", "fbd92ad5a793482aac5570387e917188", "ac81634b0e0946d690fb7d8ad7aed911", "01097fb41b9c4cbf91300e049d9f3617", "abeae554ecff4bb0ad8c38fa2829f706", "a2d84cfc801f4657bade62d42be7046f", "06a2114630234393bc0f07b3a64455f8", "c6f31bcf48de4d9fbc7f2a2d9984b247", "d5eace6b280e446ead2d1517801e4612", "e29da03f6e474f1ca97ecfa6cd09658c", "35bfe48a1b744262a8ea68bf5b5d495d", "eed4b0b927824444aa8d875281cca1c4", "e77316215dbe41daa8e89d8b2cc0f032", "11c60d67f2204518b007ef47c801fbff", "da21830428b54f76aa31b03efce202b9", "a90285006dda4eb8a3e77964294a76ea", "8164a08b9b3f49288963539305eadabe", "591e6dc92b5c44418260cf659c5807cf", "2bd3c665eb784f149dc21853103e8ff0", "1ff4bcdb97294287ae8f3e9f2dc6bafa", "bc265b1f822348469e3c8df0ea608abb", "3635aa8a2cee484b945ddf7379ff4102", "94ce06b3df27425eb8ae1a0aade4244d", "e8ab0bcb4d7f4a298b2b3555d866a11f", "5570fad8913d4bb495681b5e1dbe3950", "47735f6c830149db965660d6b2f200d7", "61d980530bf0408c8e2ed9a7997ba615", "8a9f0a924d53496c8a8f228738ec140d", "60ab33e3e0d84395a1269604f0fae91f", "64d829532bb94214b805c2de4cf529cc", "438c02b8134e45d5b2760b2e1f72f004", "0f09296d37ee44b89871fa22cdd0127f", "0fd475cb9e064d10a8ed031957cf2044", "d1d2687d51a4442d8555ed4071837da4", "0d80cc1f4fcc49d59e3a80862678fd86", "9e556858d4a44b5bbc3e3af87d138a55", "d86eb0ba47884ab081128a8761e9654b", "620e2fa48504435a85d95c2d4b264b6e", "af9af54477cc4972bf0f0a99c1344974", "0b5d06cb83334b53a91c929f8e308543", "8b9f3c47205d474b97efc6e9c6fb5f68", "b63d8ddcad7745f3b4d7e683d23f393a", "b83993ef787047cb9a31652fdd7f9ee7", "f4ee33b4a2d145bab3c5c2e14c73a3f8", "bebb055c0ac14d59a8b617399d60e602", "888e04dcb56f4c43954d49d3e392ab25", "6058ac7bc0b345ee8f5d2f631b7b6940", "ec94298c63ab4b0e83c074a9d2ed4fc9", "0d56de85d9244f38a8a0b3d84ee5d7da", "b0053efc5ecc411f902fcf3b19cd362e", "1d69552268b74bfb824c4f783e362949", "de35e43c6aea490b917084a93c4571fb", "8fdc7615fa0b412283eb3beb36b97872", "d5542140ebd34dfaa8c66f2f3e48fe92", "9eaafbfddcec4cdb998770d2cefc8fb7", "a9e2ae7f987b4d9d9636c3963530d8ed", "283159e7918540efa39d255f475dd984", "2f5aa471e247475691da674db1d8514c", "c67e91be78c74ba9b816e40dd5c181ae", "d02de13e2d7843298e61b8f47d8dee33", "76d9a70692e34521a609b337755d9901", "8a1465ca8728490dba4fd79730ea6a30", "3d563f4f9d28464788cab663cc814cf4", "6fe391246afa49e08eec5793a97690db", "47cc4883459449af8f9b35cd74b84002", "69da0360a8ae4737b6e1af2e790f2b85", "d7cc01fb605b4dd58cac287c36b6afea", "638f9ac5607b42d3a467295cb8f7c50d", "8e885a254c6f4311a3774d816e5ef5ac", "580addd60f83499386b626c6440c6fca", "6bb5cbb9ce7645c4ad45cdf056af0445", "2e3ea9571d364a40aa9917a6f49b45f7", "665f8e9a73b94639aa42743d16726a96", "34a2309f1b78432ab51a87c964946da8", "078db166712a4daba8e99ccdf44eb16f", "dde4b35063b64a28a2fd5412eb9474f0", "4d9c96f9caa54ac69dbee2755cfd804d", "aa9d3e82fed541cca0fffe35b55aaabf", "3221446fbbc24420a923884c67e0b87c", "abf0151dabeb49eab089f921c8f364b5", "488797401b9e419ab393ad5b2438039e", "4394b231af2f4e1499a308c93b0ff951", "4f1a59e4dff4470fb123ab315ded6e4f", "5ead730bc1c34a28b5b046ae270d04e6", "13b77ec58e65475b95d0041f90639e9a", "deb2a67f32e64ebf87758c3ace7916e8", "c594b48ac5b347f78c099d581dc4cd96", "68fd129101844ab18b2b107778873d54", "d7e07795e63c4ab78ca92961ba089b07", "d3450ca5684b4a5680c114d29a7ce8f5", "61ed075433b94feda586eec035251768", "c46be587d0fe4ef0b364f822f5ff903d", "c7f7e4ee797749c0939c9a3926937b41", "2a4d28248796477994a17db0fb8485dc", "0dbdea008e964ad887f112336be78449", "47977de9d961442682805f37f7217387", "241f6ce34fd242ca9a46f8232a9fb838", "846eac2286dd4d6991f80b6ae03ce804", "72858c26d6154ef3b8f90ffb0339781e", "4085f5c89ca94a31b346452cd8009dad", "47c175c19d6b4268a2ade2966327de78", "f3dc7118a9fa4b188cc3a9aaf366b125", "9f07e155db5a473abdd7b7ae0617e770", "05feab0298c54df4a2372152d4f3a891", "abeb4c3be90344469c004e29465e1580", "540e6158656d412591a5442eb89d1e65", "51a0e1eb84eb4dd6a95f30268541ccbc", "fe7d093f30854ee1b66f080c5b8fb68b", "880e13da115f4e2e9d413b75b0eecdcb"]} id="d0b36e7eff50657c" libroFormatter="formatter-string" outputId="4198784f-15c6-4812-f96a-c3f62914dbbb"
# boolq
boolq_dataset = (
    load_dataset("super_glue", "boolq")
    .map(
        lambda x: {
            "input": f"{x['passage']}\nQuestion: {x['question']}\nA. Yes\nB. No\nAnswer:",
            # 0 - False
            # 1 - True
            "output": ["B", "A"][int(x["label"])],
            "task_name": "boolq",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("boolq example: ")
print(boolq_dataset["train"][0])

# multirc
multirc_dataset = (
    load_dataset("super_glue", "multirc")
    .map(
        lambda x: {
            "input": (
                f"{x['paragraph']}\nQuestion: {x['question']}\nAnswer: {x['answer']}\nIs it"
                " true?\nA. Yes\nB. No\nAnswer:"
            ),
            # 0 - False
            # 1 - True
            "output": ["B", "A"][int(x["label"])],
            "task_name": "multirc",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("multirc example: ")
print(multirc_dataset["train"][0])

# rte
rte_dataset = (
    load_dataset("super_glue", "rte")
    .map(
        lambda x: {
            "input": (
                f"{x['premise']}\n{x['hypothesis']}\nIs the sentence below entailed by the"
                " sentence above?\nA. Yes\nB. No\nAnswer:"
            ),
            # 0 - entailment
            # 1 - not_entailment
            "output": ["A", "B"][int(x["label"])],
            "task_name": "rte",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("rte example: ")
print(rte_dataset["train"][0])

# wic
wic_dataset = (
    load_dataset("super_glue", "wic")
    .map(
        lambda x: {
            "input": (
                f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}\nAre '{x['word']}'"
                " in the above two sentences the same?\nA. Yes\nB. No\nAnswer:"
            ),
            # 0 - False
            # 1 - True
            "output": ["B", "A"][int(x["label"])],
            "task_name": "wic",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("wic example: ")
print(wic_dataset["train"][0])

# %% id="9fca2225-aaee-47aa-957a-5f8ed3177cdb" libroFormatter="formatter-string"
# define a task2id map
TASK2ID = {
    "boolq": 0,
    "multirc": 1,
    "rte": 2,
    "wic": 3,
}


def tokenize(examples):
    inputs, targets = examples["input"], examples["output"]
    features = tokenizer(inputs, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    features["labels"] = labels
    features["task_ids"] = torch.tensor([[TASK2ID[t]] for t in examples["task_name"]]).long()
    return features


# %% id="0bf6c31c-73cd-4eed-931b-0cad5d7290fb" libroFormatter="formatter-string"
def get_superglue_dataset(
    split="train",
    n_samples=500,
):
    ds = concatenate_datasets(
        [
            boolq_dataset[split].shuffle().select(range(n_samples)),
            multirc_dataset[split].shuffle().select(range(n_samples)),
            rte_dataset[split].shuffle().select(range(n_samples)),
            wic_dataset[split].shuffle().select(range(n_samples)),
        ]
    )
    ds = ds.map(
        tokenize,
        batched=True,
        remove_columns=["input", "output", "task_name"],
        load_from_cache_file=False,
    )
    return ds


# %% [markdown] id="oNvh2WGlLo4z" libroFormatter="formatter-string"
# As a toy example, we only select 1,000 from each subdataset for training and 100 each for eval.

# %% libroFormatter="formatter-string"
superglue_train_dataset = get_superglue_dataset(split="train", n_samples=1000)
superglue_eval_dataset = get_superglue_dataset(split="test", n_samples=100)


# %% [markdown] id="550abf92-b8ea-424b-aba0-10d8da941297" libroFormatter="formatter-string"
# ## Train and evaluate

# %% colab={"base_uri": "https://localhost:8080/", "height": 134} id="b48135d6-0d83-4e8a-b1f0-c292663c84ec" libroFormatter="formatter-string" outputId="362dbaae-4a43-423b-d0d1-39839d721177"
# training and evaluation
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = [[i for i in seq if i != -100] for seq in preds]
    labels = [[i for i in seq if i != -100] for seq in labels]
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    correct = 0
    total = 0
    for pred, true in zip(preds, labels):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total
    return {"accuracy": accuracy}


training_args = Seq2SeqTrainingArguments(
    "output",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    num_train_epochs=num_epochs,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    report_to=[],
    predict_with_generate=True,
    generation_max_length=2,
    remove_unused_columns=False,
)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=superglue_train_dataset,
    eval_dataset=superglue_eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# %% id="6bdad1ae-ecf0-4df1-bf27-6474c48c16be" libroFormatter="formatter-string"
# saving model
model_name_or_path = "google/flan-t5-xl"
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

# %% id="9e539e3d-0944-4f81-bcd6-dec16c66eeea" libroFormatter="formatter-string"
# !ls -lh $peft_model_id

# %% [markdown] id="52e37b18-d760-4af7-b1ab-40a6880eb92d" libroFormatter="formatter-string"
# ## Load and infer

# %% libroFormatter="formatter-string"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% id="70490c8e-d301-4e4e-a5a3-de87b53f1942" libroFormatter="formatter-string"
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.to(device)
model = model.eval()

# %% id="80ae512b-545d-470d-874d-969dba42055b" libroFormatter="formatter-string"
i = 5
inputs = tokenizer(rte_dataset["validation"]["input"][i], return_tensors="pt")
inputs["task_ids"] = torch.LongTensor([TASK2ID["rte"]])
inputs = {k: v.to(device) for k, v in inputs.items()}
print(rte_dataset["validation"]["input"][i])
print(rte_dataset["validation"]["output"][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2)
    print(outputs[0])
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

# %% id="82ffbaf4-6ff1-41bd-8665-c51c402a45bc" libroFormatter="formatter-string"
