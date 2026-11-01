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

# %% [markdown]
# This notebook shows how to use the adapter merging methods from `peft` and apply them image generation models using `diffusers`.

# %% [markdown] id="QaEZ3dPgGtza"
# ## Turn `diffusers` LoRA checkpoints into `PeftModel`

# %% colab={"base_uri": "https://localhost:8080/"} id="KwvBjN-e62ts" outputId="19c58dc7-95db-49c6-beb8-3ade1a8fe284"
# !pip install diffusers accelerate transformers -U -q
# !pip install git+https://github.com/huggingface/peft -q

# %% id="5S64sUQhJqB3"
from google.colab import userdata
TOKEN = userdata.get("HF_TOKEN")

# %% colab={"base_uri": "https://localhost:8080/", "height": 150, "referenced_widgets": ["c9f764b5036042af9f1505e3729cbc32", "b999b3e3af3744a79c0c90657ef37d4e", "f3baef4fbf4b4ec08480522be921f841", "bd2740e191a74558a77a965b4f2d7f28", "6f6a6cfd50404f1ea09f83e95b04550a", "713dec1904ce46f6b2d5a9b7e3e0373a", "a6bb8206de044c74a03d1a64c801e742", "f2c67c29e1224df3b2def5a87eb8d368", "486282a4ead148868005c592d74a4ed4", "48f51c96b7574946bf9542633eb39135", "35c810f4bfe741f091f172cede413950", "67567eea233b423c8acd62773a4adb30", "5161248cd0384d5887ed231ecd48c82e", "6f72ea2c284e4e40899375c7b07c517f", "d99a364420454ba5bfd510d1226b94af", "d33c90e8ea0945d397659f6a90cf51b6", "097fa44254ea4ecda7c8db995f370afc", "9fb982789fbc4582bceb356e351db438", "0204153b17ca4bb0b21fc033393ce9bd", "f02041b1d5e1485bb2ba02b00fc2c242", "4d327c9e91b34c7b84cedd8f9660e9fd", "b44f7154c55146a3bf5f4bd9e438086f", "b27ac2aaff694dd5999ab2cba91195da", "556730e12d5d4e0ea51a0dd1b5aac331", "584dd148b2344bdc92f1d0850399aed7", "ccc5bdc185a84901994577ff7f1bc962", "2c75632d8fdc4458814055172d1d72c3", "2f016774e7854ef589442734d0bb2f08", "a15c3a32bfd347a98c2a50d27cd5b9f9", "67ef71b6521b47dd90e3dc0fd03016f2", "989c54778eb7469fb91e4337d6f49b0b", "439cf6c6f9e845cea4008e3219454ff4", "460c3c96d1724713b78bddcfc1f3eb97"]} id="1YH9xWDcyhaa" outputId="8b7b32f4-4b77-411e-f499-7b2cf7650613"
from diffusers import UNet2DConditionModel
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

# %% id="-kfTBaLR1Mp-"
# So that we can populate it later.
import copy

sdxl_unet = copy.deepcopy(unet)

# %% colab={"base_uri": "https://localhost:8080/", "height": 593, "referenced_widgets": ["0b7ffee735044ece90f010c6771d2d69", "27671f87d0f3400b89e34bc428daa53c", "33a018bf2ed547caa69f8ca91dcbb112", "a91df5ceea8647d98f4d49701bb00969", "c8555cb30ade40b9bd9a3435d6deeb67", "b28a2b59aab9478e9ac3ab24a1de6f9a", "c08fca70486040069d4f8f1df46a1074", "bab2458ba3f54b229828ad9c8706aba4", "2a0f746d6eab4680b6c44ec5cdbd8fd9", "5269d405725946b3a657d3a8d7b25885", "b4a697d8335c435ab21219716a1da022", "85f22e78d47641d19efb6e6d62f6a014", "d8bbb7402f3e44b2899fc98f02cee87e", "8b41763280a048d485f06682ddc12ca2", "05791b70a24a42138755141602399c47", "753f64d9069640d985f399a058fc9b5b", "19679186751b42e3ad2c44ea46c82a9c", "d4baf7891f854c9c9898314633f97356", "9c6ab017f5fb46eda8e3c22e9dd2b838", "10bc970e474a46249e2a5e9e43fec7be", "5c838962c7854541b61877c6d42481c4", "4d54c07dd8e94557b82521772c1a2825", "efd33121bdfc4ce195a07c9ef523a477", "d2cd2572973249baa8e130b5777f4147", "4e77d8b6cdb94fc68974d27b24cfb3fd", "c292a598350d4dd4bb9b70aab1320c29", "c68a5ad3bb664e8785e724085d208e96", "b27c74f0c7bf493fa8bcb4c5b9c9c100", "09638c8da74f4dddaf1d5d94dd8ad885", "6a8a634cd3844fa081a4622d224ec940", "2460b9e05b58481e898b139b94532c14", "cdd8f9b1592842b48e1ffa80f8ec8246", "ba9b002888b448738ba4b127e1046f5d", "54ab158c643540abb8b3a96c1ae3ecab", "55130e37444844f88286a87c8d153eba", "7e3048e0fac94dfdaca1bc862ecfce15", "643ba607008547ba9572fa4880a7d0f2", "5738e5d103254c5485856f97d82954ab", "beedc3b275c24432a2e959dcf9dd418d", "10690b309908402289e9891203714199", "0c68c6c65fb94650b15069a25d9e1699", "ad233dfd52034b9e8c6c2c5b86995717", "2ef25bd6dd644347baac12366d7002fd", "0918b23f41e0404e82fccd08cadc6ccf", "b25e822b3f77431fb72b4780067c90d9", "55177afb435f44898df300143703e4d8", "b79d3f5bd8024451bc7148ba2a5029bd", "eacd646e2b984e60ab603bfc6d631de8", "d158a80f4186411a8a9c335a8d5a888e", "7049676db714446b98bba16b5f1b049e", "09172ff4be4e433483d27b576464d1df", "ea257c1c73524141b87ab3c1ef85c908", "c474ed5e340146baae6c38f62013afe3", "2ece04bc10934b3cb9d383abfc5ccd6e", "67a03631ebc54f99928f0feb18ab38af", "40ff502d2d5c40378c25cf84dd3323c8", "230bd59922e84b90b0a141b3ac1e681b", "6d0f946764444df28cb0da0fd0a408eb", "c995197e66e04874a9f5d34db98b8890", "f2596717405c40e1a39b721386e7a972", "f8e6babf4fdd4c8e80d6dd24ff22d464", "5e030e8a026b4513aa954203169f0a27", "f9ca5d4810b34938b6f997ff66a8d541", "b418ff1733db4efbab1b00b632b894e2", "83f2d1dfaba54da38f0421b69930c3c1", "577cc4b4f27941189c62951046db24ec", "a16016987b6145b69caaac6712d72835", "c82cf8cf90ed4e93bccffcf75881a56d", "5a4c6dc09a1049c0adfca9834b045a25", "c4d83c2b37504473afe63209a178b4cd", "fae9d16daace412492b048b012b8d6dc", "421b59f4021d4c4d930c51b6b4c7071f", "60cc7415644d4e16b88f8fc5896b4b3b", "fd13a58d6b444a0f955832647f64df12", "8d492d53eb7f4225b516a65ef80f24e5", "70849993c0c94ecf87c56e430f06181d", "e820c557697648378966ed0a073826c8", "a9ac3b2188594c19885bcdb9a659ecde", "a708db5805424449afa269b714ebb3b3", "6d492ddabcbe4d65b5d311834865ab92", "e17e3253c16743a29f09b82d23c3b26d", "e67a69c294334b01974f8bef36f133a0", "24cbd9338ad94801aa11f4dcb2a867cb", "c88fcf0d20154b1fb9b7e8a00116b5e6", "ee8407365f5d42d9b98536152c9efe92", "5ceb5c369f974b84ba850c9e81730a0e", "a7767d5a440f4c819cdb87414f2187ff", "f0bc6b14a299445ca705b888b3047064", "fd0ff16b68b2488d8c31ebe700dee9c9", "0e49d820da754da785bec2e5940eb9f6", "0b53b908088648e3b2beadaeba0f5da1", "804e7ee768794bba88aec3137f418868", "0d786d8386ba49d0b53a5452d52e722d", "33d459c0fbfe45389abe7eb43d2710a1", "dd42a2ff90854b74ba5fde1de26b4e15", "a3ce5829f8a640e79a19d737663b8474", "62afb52a01924566b52f6c2d9ffb76f4", "7a52c380a91c4f49bbfde658550248d3", "3e418e46c47841ac9d717a6981807f68", "8840288388ae4162884feef9c8e776f7", "67b7783351d340d99c44149635b9be84", "0f60c0f123954744ad13b670ca6dce77", "dfabe7aa70024d1aa868ef5e6650dc6d", "7d282fda276343c3aff99b001253ecc1", "4cae160456c74a5fa761f026d64b2e35", "819a26acd18f443882feb129f2c576d6", "d2ed1988cf8c4bcdb1793b3d5068efed", "6f113303101b4f448380a878d8900bf6", "60e8e9f8e6ce40dd910ce1a9410b5e24", "6703a0417c474db5bf261fe8679051e9", "cb89ecd5a8c14051985495da1797a202", "61db57127c5845759360bdf8b29dac2d", "ab9c869439a94bddbbc0c6098f4c5b2a", "587bd0b90afa450ba82e49cf86ee135d", "955ec2551f8b400dbbdba68d7449de76", "f46df85f441e4ada831f0e2b142f296a", "d4f1dbe4ce244abc987b1089876e080f", "13ac509e0d5f42e9bbb6deda62f77923", "4a4a50a17c014f189b52119901104d79", "cf42fc299989442f94f4a9df63005ab4", "90b9300ba00e41e58170ec5634622985", "4fa4f545258b4d6ca014a4c84ec4b24a", "993e5303f107468f83dbf51026e64301", "1e6c1c848e364ce4841ddcaa1383cfba", "d432098a941c463599648b156abea24b", "917139ad07a64e0cae89f2beeffae956", "83afce4becac4f37ba916bf1901346ff", "071e40d45ad14fc19b1480927d15d2ae", "fe19dcea6d9a44d28f077e065f1671c4", "08fddacdff4b4ae09adb5440fd86ae86", "d62465d39d7e4265832901e9b9707993", "4f56214f69034077bafdfdabc1c2aebf", "29b67774c6944bc7ab93e7ba0eaf867f", "0805d34df44f451d9b9910dbe5999245", "2f2c0ea0fc914e7981e34d01751f74d8", "5f52ae61812544f29f774f80fcb7a09c", "ca5dc8342ef946a49d9e67a21f1a67c8", "babce0e85baf4279ae1d22d64006667f", "5cf6cd09cda64d688f4a0f7c511533d0", "f668dd13af6f41d8be358f7db5261c54", "8a8ef60b3b72452fb9ccc31052ab3b4f", "7d29b20296aa4e33837b8ad53fc4adba", "6a94d1d176b844db96de0c0e3cd67701", "c80901426c87439481078b9da2e0c772", "416f31db79fd431fb8dc06e994421b70", "1971fcd35a564c449b4f437dac46058f", "76a474e4aab14a5987cf25d1391d578c", "ff3bf3f1873c4b01b0a547dfe02923ce", "a6295c7e7630444c9b7425b884ad9707", "3ede828a0374453e9aac9a6695befae3", "dc4391fe30694a788134fffdd2a23d1a", "cfc6ec59d45d42d187f42061902abbfb", "68822f8f861a48859df630fad63f51c1", "2671327d35e64f0da670a3a611fd0886", "aae194e7f71f4eba80ffd18f91f083a8", "9ad4192f6f244264aede1b3c8ac3c57a", "90558769855c4bb08ff2fe4af940c45b", "75384779c5d540d28afb53a0e318b674", "a287f197e4fe4f908e3ee0a0e6cb35cf", "fdb799739700447d8a5198f1f4f9b17f", "683db0ff105a47038c350cfc74b88345", "b79d32186443469d94836b663bf156b5", "86cb336c1f684867bd13dae0370b4d36", "7f63b3d80ecb4c3eb837bd0e616e1623", "4f6e7b21ba0747f19b9d63468861c988", "766567be45eb4db6b8d7a3364566c1dc", "3c222a56d4404863a1dac60f1d03835b", "db862ffbb44d450db514173df4c7f301", "ddbeb13bb8174fc0b7d5543108d1c4f5", "5b3bbc663d504fb99318ba186e6b8499", "30bfac68f4224152b048d6ccf6013c5c", "971f59d5f3e04697b3ab3c39ec0fc667", "fe958df746be4dc1871bd58628697c3c", "54417d8b9c5249d0a028d9e831dd8be6", "9a37ffb9810f482b80f245f64947c371", "bdfe4e4109a14a41bcd2e1e4242d82bb", "22e894ba852e4072a1f63a83b3a98b16", "95e24bbc8397455fabb724ed3c330511", "d35b3848508c4b6390c243866649439d", "c7cf10df8d7944aeb93947d5f4156c92", "8e3b0b9f26a34ab8a932756b32834fd0", "b1d427082b9f452eb6546c7d55016b36", "061fcc6e5f3c44c48cde212c1ba515e5", "d70508c304794bc79e80aab136eaf65a", "01b20535e40b47068723073ac6c819ee", "de0b54a59d9f47408d92915ad746cd5e", "eb9a5f255fa0447eba6a33e1c30ba166", "d1e4d2fd70e644f986104c998db7e53b", "f786a0f386f6486083c15e576f6eb3e7", "6715d91c6b62426aa49c344b65bcd8a2", "0b0a85e1133c4ca3ba716e2403511703", "017dc44dbe1c4de491e003a1e279a218", "e3a1a5e9f29d4d28b0b9496493dafa21", "c008577d922e436aabb8680ca0d13117", "bd176c410a4e48a382a1b688e77aafcb", "6b586ad7a3054c83b14243d69484127f", "34fcd527a59748cf832c9efdb954fc0e", "fd672cd5ba0c4695be4240707dcf4bf3"]} id="EMTVH9cLEZYi" outputId="7a24b4b0-71f6-4c65-a242-fb3d502da6a8"
# Load the pipeline too.
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16, unet=unet
).to("cuda")

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["5e677518051c43768dbf06243701c817", "d727913663634e368ade4a7dc64fe74b", "eb73095c804a4272856fe348fa3cb1e9", "7e9b46b10fa24dfea489dfbc150d2a2e", "c8156b0cc68e4b3693dcabc530a4ea9a", "d987907f09084d44b452f939aadff65e", "e976b994189343f5ba7d762ef92c79e2", "9810873713024e79ae6d338dfeae5876", "2e18ce21c01a4a3ca992622957e7d297", "2db944a049a04426bba181fddb2801b1", "683863a313034025ab99fab5810f39c7"]} id="D5hL5156zPis" outputId="2510b8e7-c030-40f8-dcd2-ef76fc8529c6"
# Only UNet
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

# %% id="p-7YWoOs02La"
from peft import get_peft_model, LoraConfig

toy_peft_model = get_peft_model(
    sdxl_unet,
    pipe.unet.peft_config["toy"],
    adapter_name="toy"
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["c6bdeef396174d51af9eee277752bec7", "685797f8907c47ffab4fe7a81ca22e63", "414f8301f76043758f69bbfb6960072d", "23c0492f021a4b60b1d84b0b82d15378", "550b3ad10fcb422eb66f71ad95988616", "222da37b5af14d60814c65cdd1ea20be", "6a2b5bbd4afe4e0cabe39e95ccf528be", "7ba112dcece64386bdbac6837004891b", "2ffbf400ff1f4cc9b358bb10c6b9d99f", "5405d8a1aa1d411b895b0523fb8e4ce7", "04846af1def142ffad328d434ba228fe"]} id="a_2n4Odz2a0c" outputId="4b3b801b-649f-4c69-b75c-f800ac75c17f"
original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}

toy_peft_model.load_state_dict(original_state_dict, strict=True)
toy_peft_model.push_to_hub("toy_peft_model-new", token=TOKEN)

# %% id="z1DWL0X12rxD"
pipe.delete_adapters("toy")
sdxl_unet.delete_adapters("toy")

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["d18c97fe685e4be080125ae770526255", "e44c018bc76c48cd8738fee5966767ce", "57b491647f1e49cea7ce34774a963936", "484f694f734b4a92a14ecc4d048db0af", "bcb3ec98d25b4c138c5e8f84c1e937c6", "b0abce2d8a2046dba320e788e33e9d66", "aef81ec1a7e844f883beba8c5754a8af", "c040aa1f65514be28f0ca8ecdd1f69e4", "eec76868d92f45d7a6db2e232a45e0c2", "157c0c1e85ef40fb99e6cfe0e176be38", "99717def9b6b4afe8a411a3bb83320c9"]} id="9PW-SfwH5L7e" outputId="90721ffa-faa5-4628-994a-7b719a4ef02c"
pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters(adapter_names="pixel")

# %% colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["adb2daf9d62f49f8ab1f0144b717d41e", "bba38c266ceb4f30bb4bc1eaf5e3aa96", "6926c8dd4e4e46d089bb387333691df7", "abf2248c725b4837b5c2babef7f4ff3e", "39e4bdf8d621451f984f6e7302fd6961", "4abac6a83641414499f9b6b1514d1695", "bf6e103b43844f17968eadf223a42acb", "adf53d97af214cceaae895c8abfbd909", "9d74948f7bcb498c9295e41e690a0a8d", "d872c3900b8b4275b2224c9ec5e7d78f", "4fbaa1bd51bf4b1e90337a435604d2bd"]} id="jHSb-iIf7IEb" outputId="29124d4c-b58f-4f0e-c59b-d79b44cb162f"
pixel_peft_model = get_peft_model(
    sdxl_unet,
    pipe.unet.peft_config["pixel"],
    adapter_name="pixel"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
pixel_peft_model.load_state_dict(original_state_dict, strict=True)
pixel_peft_model.push_to_hub("pixel_peft_model-new", token=TOKEN)

# %% id="yoPzMtyqG2ZO"
del pipe, sdxl_unet, toy_peft_model, pixel_peft_model

# %% [markdown] id="Zis3zKZpGy8w"
# ## Weighted adapter inference

# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["d74b1c667d42472b865ee1cbefc33a60", "1074a274530e46c5b5a3e43653da43d0", "dfde3984062442869bc8091bb94f2c36", "a6f2ce8830734be4b8efe5ca0e14e990", "07da49088e8d480fad03a2e828357872", "5dd09616ba6544c499271054e8d8d2c5", "fdb1fd279a0241429e5721ae2e92d217", "7c6c4dff0a814bc6a7ac677980b45add", "9a9da0d0e3d84a19b5d188c9bd6a83bb", "de31002ed7fc475c915b4a29253108af", "fbf3d268f30344b7864ce691d5bcb1f3", "153a93d930df4ee396e03f1aaa6f04f1", "c8e40d44aeac47a78f6502771f1471b7", "270ddad8d7704f929a28c9fbbdabfa26", "e57d317b3dda43bba13ecd4514f776d3", "22b55b5ee1af4ea7b1acfe511b194cd8", "a48d3ad24e9744f7898d1f2c5a696ea2", "f29673e57d174839a0bde70bfa165715", "13f5160dd981465890ada8a2cef22d5e", "cc06ed7338b74ae6a1c563212aeb9f94", "3a017f1d0ebf4a4aab57ac3eb0788774", "48643ea67f2f4762bcd27de1d4cf0fd2", "5c3d142907404cef8fd624839a166530", "abaecfe7f39a43bb8fbd655d1a3009f4", "d488374da5e74ec3a5973003590a7d69", "5d8ae9661dce4a1aac8d418d84f3a209", "32132ecbe71a4c2c903932513c2a1aa0", "05700817adac4fdda6c95dd00eaeae38", "5a7d87338ddc45df84080e0096a31631", "cb9a536bc56f4d0ebf285e7f73d4730e", "49bb475a11104e9496f2623e3d5caebd", "887c5eae5b154eccb4a1caa3deef6e94", "e87dffe17f1948e9ba794eddb605a908", "2c352a90375443da835eef55b3e63303", "db8fd6b2687c449fa0600d3e87c96999", "17691a346ca5407a99a5e385450c97eb", "0bca833a6aa74ebaa8f69feb738806bf", "6bdb9b0b68c24b84a748c18ed927a8d3", "355efc45ddaf42498d72d6134a28c87b", "8b6464ce614c4aa29ac66ecce29b6cbf", "bc07cdaad5b64fb3b0e1f8c214bba813", "9e2e87c131a140a2a37dfdf483d27ced", "47468a75637d436f849283c295a74ab6", "af5003cf40ae4dfaa0660f247598856e"]} id="gEqT1vFtG0_e" outputId="282ce865-c653-4912-e497-ff825c896ae7"
from peft import PeftModel

base_unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

toy_id = "sayakpaul/toy_peft_model-new"
model = PeftModel.from_pretrained(base_unet, toy_id, use_safetensors=True, subfolder="toy", adapter_name="toy")
model.load_adapter("sayakpaul/pixel_peft_model-new", use_safetensors=True, subfolder="pixel", adapter_name="pixel")

# https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter
model.add_weighted_adapter(
    adapters=["toy", "pixel"],
    weights=[0.7, 0.3],
    combination_type="linear",
    adapter_name="toy-pixel"
)
model.set_adapters("toy-pixel")

# %% colab={"base_uri": "https://localhost:8080/", "height": 186} id="QStyurhKsP_g" outputId="5e3a2627-27a2-4771-e62f-1b81ded2b87e"
type(model.base_model.model)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["28a5059a2cc445d783e94ad5d83a0748", "faddc146c69545cdaeb81edc8a0cda70", "cd08b99de03c483a965248ff4df752ba", "d37c6ac25fd34a65bf307896443a5063", "66da596ae59d474f8a82a600174adff6", "b600178b161d4a87a0b832a169d7caf2", "b3eaa188cc2e48d081488eea4ed2971f", "c05842ed12c848c68c4a69de9aa742a8", "18e3ba4a61784f04b3238cf273c90c3e", "42a623abc5d84c0eb7de4e5323bb6546", "a65484354d254516936fcb425917a4b7", "64029a1e70e040b49e38d38bd36823fd", "3e1ec3a51e9b4fbbab489d34640cda90", "ea0910fc31e44597968b2129272cc94d", "3a0e9adc345f409cbcd79d1bd19219e6", "3fe7e7f00ca746cc8cc762da6f365fde", "6466eff2786241eeb142f17758894bb2", "c181408b9b2b437ca11d584e3d1e94e7", "0371aa5607604c06a868deb2a413cb31", "c212598c5a8747f783a6efc18816e868", "89ed71090d5c4366b21d25ad102e7da7", "4ab21d7b956e46348ad6f6542fd92c2d"]} id="iHwVV8f6s1EC" outputId="47cb80da-266e-40c2-cfc1-3f3e5421b50b"
model = model.to(dtype=torch.float16, device="cuda")

pipe = DiffusionPipeline.from_pretrained(
    model_id, unet=model, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image

# %% id="adLnc7sMRZlq"
del pipe

# %% id="nIwIQK5zRX25"
base_unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

toy_id = "sayakpaul/toy_peft_model-new"
model = PeftModel.from_pretrained(base_unet, toy_id, use_safetensors=True, subfolder="toy", adapter_name="toy")
model.load_adapter("sayakpaul/pixel_peft_model-new", use_safetensors=True, subfolder="pixel", adapter_name="pixel")

# https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter
model.add_weighted_adapter(
    adapters=["toy", "pixel"],
    weights=[0.5, 0.5],
    combination_type="cat",
    adapter_name="toy-pixel"
)
model.set_adapters("toy-pixel")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["65bf3df199b44763aa223fee96889e17", "a32666723b0e4b9883a78f56295c4356", "c08373a044204308ac882dd8cf9cdd3e", "fa7876ade8e240fc89a35a1f8c7c7d3c", "de583920d3b54774a486aef4c052e50d", "979c68bd2e224a40b939c27f32c25dac", "0027f3aa006d4276983691ad985ce91b", "a0f8a1a3512443ac84799b95da70ca26", "e08d739f59874064994212363a307f6e", "52c7e22284b0468c8bc0c3b1cad047fb", "17c29bbcbc0c437c9c8bc83e0b085f1a", "65b15618abfd4e6f9fafd64813e86ace", "1e8747251a1f4cca970857911d1c4a98", "caa46820018f47abab4a962afe51cc34", "22a2ea45880f4b0da47f1b213882dcb0", "0cf822f588244e54b5264176f9611164", "dd9666d76af04b72b08f59023eb04ee3", "2702528a2ca049fc800ad44c492690ae", "aed53b6480de4dd4bc7463af04840952", "447d29db05384c55a57c5fb1bd121af4", "2d7e7e816a63428b8f30471a12b57bc4", "e15aab8dd01f4d5582db80e6ad9931fc"]} id="29iGITdnRhFG" outputId="dc6a1e54-3f76-457e-da1b-e8677be5c31f"
model = model.to(dtype=torch.float16, device="cuda")

pipe = DiffusionPipeline.from_pretrained(
   model_id, unet=model, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["8b88a1a63cf242de8a68962f50498c72", "777c5583820248838f0c39e82362d9e3", "491da86a94c44b7c87366b4bb72a3bd1", "31c3a375a2964a06a2473925fe9b197b", "0ba0a0fca31c482bb628a6739d341601", "734e9a834ff74b64b17453013208a116", "9784d9210d6e4214b78ab5f8c33e8044", "490be826ebc14d5ab4e9f0e10ec79d5f", "64ea3c5cc12841f59d1acba12deb6a88", "fc1391aaeaad4eecad967e800a669ec1", "272fceb4e9484b389067080872b1abf9", "5ab5097f19cf4474945f96741c444d71", "6d8fdd0303774305ae20dd39e2a1706c", "031a326124f1496abe1f3bf8de720029", "b9feee6f48bd49209e72e5c5e3136f67", "a2d497a2ddb04d8fac8a4c0f8aa7f5dc", "7bb3f4cda33947138c61aac74d952289", "d3b73a841e68425994632d0f05cf4f16", "71ad1e3dbe44437d8df985cfae207dcd", "598fa824f517445394d08c37393f9f3d", "daad0d12aff8470d990fbbbbe19d5891", "16e18e872ab64b3f8bfb32f580c3371b"]} id="sQOnSrteuS-S" outputId="44c5e61a-370b-44bf-a5e5-80b7787088e5"
del pipe

pipe = DiffusionPipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
