# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: 'Python 3.10.11 (''accelerate'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Finetuning Whisper-large-V2 on Colab using PEFT-Lora + BNB INT8 training

# %% [markdown]
# In this Colab, we present a step-by-step guide on how to fine-tune Whisper for any multilingual ASR dataset using Hugging Face 🤗 Transformers and 🤗 PEFT. Using 🤗 PEFT and `bitsandbytes`, you can train the `whisper-large-v2` seamlessly on a colab with T4 GPU (16 GB VRAM). In this notebook, with most parts from [fine_tune_whisper.ipynb](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=BRdrdFIeU78w) is adapted to train using PEFT LoRA+BNB INT8.
#
# For more details on model, datasets and metrics, refer blog [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)
#
#

# %% [markdown]
# ## Inital Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="eJrPyQM5Xhv5" outputId="cfd6d8c9-964c-492b-b641-8e80e337f783"
# !add-apt-repository -y ppa:jonathonf/ffmpeg-4
# !apt update
# !apt install -y ffmpeg

# %% colab={"base_uri": "https://localhost:8080/"} id="r_Ivl7qlX0dz" outputId="2caa9eed-f01a-4603-a527-fe3b0b58b6e2"
# !pip install datasets>=2.6.1
# !pip install git+https://github.com/huggingface/transformers
# !pip install librosa
# !pip install evaluate>=0.30
# !pip install jiwer
# !pip install gradio
# !pip install -q bitsandbytes datasets accelerate
# !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main

# %% [markdown]
# Linking the notebook to the Hub is straightforward - it simply requires entering your Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

# %% colab={"base_uri": "https://localhost:8080/", "height": 303, "referenced_widgets": ["c60690c2aee74763bf23115553f4e640", "7d3d6c198e794219ab5db59f0228c8ab", "5d5ea0207c6148769ad9f15b7b3dd92d", "642e28d258ca4c30a5df94c5cf7e0471", "170ee581427d4f30925dc393d124c1be", "d5d5aa24182a4e04b3fdae1ca7fad52a", "6dba643113a547ac9b6e121d008791d6", "cd9bda1053a14890ad9091c63c0a0acf", "4522666ebbcf4647b06ce81a6316fbbb", "989b3df296504f34a62d31ca0d6d88bb", "e8a7a34c6fb146f0b38a40f389a617fa", "960553d142c446cd8852523887a5cc04", "441820fb176048109e0f8f7e9519d735", "0bb38c654e18429a8396466ebab84504", "1acfc4a2809e41dd995817c3526650dd", "6d5801774beb4b529b227ef2f098614e", "dfdf23cde48c421caebb573060641d6a"]} id="ed0OpduhX2JF" outputId="ecc2048a-b46a-4b20-b94a-5912924feb3d"
from huggingface_hub import notebook_login

notebook_login()

# %% id="e1da5fff"
# Select CUDA device index
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name_or_path = "openai/whisper-large-v2"
language = "Marathi"
language_abbr = "mr"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"

# %% [markdown]
# ## Load Dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["020176f5aa0a4d489d022ef5e41ef3f6", "cbbba08d9e634560a6c0429e32166fe5", "6d9d1609edc8471dafe653e4da32eeeb", "435b5708e6cf487ebb1799c7b64a8218", "931b96b21b39482298f40449424fdd34", "79da882ea494477a8c94fdac7acd644e", "f14847261aa247fb9561373e0495f3e5", "ce8f306e745d4b158c58058d471de037", "0bd122731d674cdab803a9981eb28237", "46211764bc6641b4b693c16c90747021", "f1884e5a392941bfa8c484496d87084c", "6b1cdae6f5e34d1d9a0b5e7adf5183db", "4d29aa0885214a2aa79701cc226edaef", "5411f018d4464e839f2fb21ad3f026aa", "5fb8e97b25b44e0dae9715eeb97acc5d", "783d4d18627646648f8c120b728babe1", "1a721ed289684104b0fbdc8147c2311b", "4880366554ad4f6687a25b5a17877dcf", "283f5528547b457698ced126c25c2a44", "7c90db94a11e4a5aa432e664b2af4e7e", "7f6a466819bb45e880eb989f388a7523", "d96245da43944c4b8235e0cd02c1aa4c", "41187d9a120448fab6c8608f226971fa", "1d4bd11921d145c7bfd1a8ce0700a666", "2853a43036244f54b74db99311bc580d", "d9a85d7c76b54199bbf7646448e3458c", "b09d153958cf4a28baad268bbda78236", "003227997471488dae9ae26dcbff89ea", "deb18822d58b4b60bb75460f0a5fe921", "e6cf97ef7bc541d0b9c6de206a3a45b0", "a4b16b5279504dd694090798f5925d65", "564cd321c06440e9856f10f5c40c20be", "8a91574c4b6e4745b2b65885323b4d25", "9202065d8e6f425d88e4514dde70992d", "f7b4ec74e2ac45bbbf0265d0363f4d9f", "3a083523ae604362901e4e31c39fe949", "d1bc2ce48c3e481b9059e882b5102946", "30c688df949042ce89337643f6178230", "2bc85a5bde9a454990d3bb7de5e3c7c1", "2ab45b22ce3f400a81cc451b9d7c9eb8", "cdd08679c28642a184805912c07b324e", "90744e5529c04f18b11e00326649abe5", "11b9c50e720a466aa92c64b254d40778", "8f856d6bde4041149324e1f53d19c1cb", "49545bfc91c848b8a465e13d0b45fa34", "45a85ceb24ea444ab18a6985884be966", "ac9c1141ca7c453f84a8ab62b2de9158", "bd4e3eb14252470d9c8ac60a32948a72", "75dfc66dab48405b85073317c8dff155", "b16e3a75acd0403f865849f1de6ca654", "78fb0d05929849ad9aa7a7ded63c7b6b", "3fd0fd7cbbef4785b360891c12017f48", "a9c132959dae4303bcf6015106ce3453", "2ef66808d51d440b9998064e212df420", "45478f2ce991441c8ceaef9acf745084", "c1f019686c564cca87c240a75ab71ad3", "54589117bf244027ba024ea85bd1fd77", "7b7986ad93f64956b8d198d2cb4acb60", "dbc1a016a69b4ad7811d2701a9520a2f", "72f8e7de2a5d4155a9e9146f12e14b19", "46e44f07b96d4c32bbe75bb89159f093", "90b53e96b3f04c2993969b17547ae0d5", "8843f70cfa1f45299a588e96e9c1159a", "8b2e2c650f4b4bee9e763b7c59525ec8", "fb9f013a6188463fad6db70702576c37", "93e2efbb5da747d4b94c916153ee9706", "c44f472624d84e16a0f380580d36ca61", "65188c1fdba2421ba85c2cc349709600", "0683cbffd97a4b75bd5e00a05d541fef", "4573606592ce4b2a914ed0a70b69f9af", "efb85d003fb54f55aa4eadf2ab8b1684", "a235cb3d4d424efeb30901f63fc1dbe5", "2452ab9f6f974d1d831f7c81e884d00e", "bd7da2671a22431889fbfa2ae1e0fe2c", "47ee1eec97cf4af58ed4f50944386a7e", "a05426c8e5f849b7a972dc0df3cd84ef", "1e8cdf737c93431e841e129abb541e22", "91fc5846a32f44d5bc5fa30fdcb3c638", "daf4005d1d334608846d0fb2fe4f837a", "ba48d89cab9346ed92dc338779f9f828", "6b0e0694d895445c9bf7c955a116953f", "ecb7ffba323743c68961e294e74b337b", "460be80f176849e0b1241e3a4fc18b74", "74e2bcef1ce94234bbf6ba0d6488279d", "73bde731e20d48de8ce66b9d72fb95cb", "9afdb23ba9ee47709f194e0e92de0edf", "621f989f46f84d73b96a37c954e92b8d", "89bb7bbcaa194a23bf3d908a4782ccb8", "59e79d9132964429a2a2abbdf4bdbf32", "2e5c1a371742446e8bc416d4735c9ce3", "e57b15cc74e7474083e87722d6acde47", "e899060f1edc43b980fe6f3bbe13c609", "a6963ce72cb3425791804abf1718ba90", "41a7cc33e0bc4bc5bc04dfc72bab7a81", "28279812acfb4272a1c28ed70aa1362d", "451d3851e29e4efabbc2c235dea718da", "18dfdd09a3af49f5b17cda27872d0ba3", "aaeec2b7986d493e8d238aa14f2e5937", "0894264041854eea960707529f3fb8c7", "3a4965f422f14e1ca2a63e1852235507", "e590170f306347f3a82b76a25d37b652", "384f4d9515d1431589a3cc934bcf5ea7", "edc24ce2510f45f8adde0a187016259f", "c2c2608dd091493795d975f1a3cc3762", "1dedb58d31dd43db96ddd7315bb7e2ee", "308e1ee4593b454a84681bda10921207", "8f52bab9ebd049d6a57686d621f407ac", "2958a33cff794de1aa8128327bf405f1", "1b2780d8137042449bd6779c70bf43ca", "ddc2a8e8ef4d429f95081c4c5baf1fb3", "ce06b2a0de6c4fb8bae36bc4d7f63270", "c4cca1778f314ce582bd09b9b2494f82", "3bd70d937e924f61b943acb0aaf15619", "c81c5d3a4dc5409e95a6410e67fa9857", "dd02d1b31bfd4b19ad626d6691a9b293", "b1d780c721b840d7a06661a7a5e63236", "d19402df47464044b36fb5ee4a0c1c4d", "b1221f4e0a57482682ea4bd6ae245da3", "64533507b97148008548e55623b6c2a3", "33888a5cafe6495782309dae44531dd3", "7d5ff2f1b8794bad8286e90307bd6a61", "7a2d45b371ba47a994e4372437f51cff", "dec9f287435e4c6b9fb1d1ead2ded576", "db5e1bf1871546408f233f0cbc37b136", "af44aa66372a43beb9812ad9895d8d1f", "9b427631384c47ab89ee1352c0236afd", "00071f8cf276478fb2740684552f1275", "64378b1064dc4036a9a4c8813013e210", "ed7daa32c94648d5951876229f9835b3", "eb883a37a9cf4945bd864decb4fc87ea", "7b190bb2bd234b7997ca041baddd511f", "7549fd0b38364d8580f8eb1549558a33", "5cba5542df344d54bb80dcf7be56b2ae", "8daed385e6564c7897fb6553669a9065", "25fcd4b584a143058266f7f3a650d69a", "58a15b8df2d54f89828bcd205d6bf298", "8b51633b2fe5479db0fc73cd3f0ebaea", "d2c1704e34c34d12b99c31d64fce88cc", "6bdce3b33872457ea67a3a3191f438ca", "c8ba787279ea43aa97b134c134d4f183", "6efc3be410ef4710a59dea8dbdabcdf3", "6f9d8dc63c494c76b36074af773330ce", "64c0bac85ee446e284ed85ec0eb5ad44", "5162987085ce45d88233f34ca3a41ce0", "00819c11ea23467689e78a00d89d1b09", "65dbf3c3a80b468aa8717e97c14db6a2", "8cb910ad08024c818b99c2e30e30b039", "0235efbdc6a74cce8050ab1116466427", "ee1ae17fdf4143ae8125ce5e2a7e9066", "481ebdff3ccc40fcb7a8d848b01ae8db", "6117f9e9718c4f4388bd1feec34578b4", "623d0c6427964291b5c74fb18bfb4ce4", "14bca5cf764d4464a186961ccc6bdb3d", "75c793c91b9e4aa994c2d34ebb16f7d4", "3502b5f8aeab4bfd89c83b99ab308b9e", "c43169085f2949f1b8259cd0b767d121", "4903414a865c4813a9371e8b301f1af7", "ec7e7e3811e34b4a8c8cc31cd021cf20", "a22db8523d44457bb96448d08129988f", "9980bca9b1334893bf6583c325122f50", "931fc769fbeb42bf82df6ef5f914bf48", "e944fa694d824042845364bdba72d642", "3785471f614f47c78e3000a303b11ca6", "dd634585d5e64c97881b132b1d59083e", "785f3df156b946449c492f1296656a70", "47a25de2e10d4b55a9e6827b85e49c3a", "898c7b43a5e94601bc093e0cde28c2ec", "a03430c0cdcb47bfbd2ffd754074d692", "cda6af0ca01d4053bcebe06f3c41d887", "abe561c67f1f42b29c33d4c296221b2f", "21ca204b93994e8790c9eb7b0722761f", "f628cb62f5fa446eb608467c1ecea526", "e20264d19e804f9dba3e3867ef9b31bd", "43141dcab2324fc6a12f9b8198e10154", "f773a61b1dba4e3eb0df36162efe9abc", "bfc9036c9c5f4be7ab191b92d92f1352", "0f534e081ec64883a5f88d06f93aeb00", "3a94afa1e03544f68df7752a4f503fbc", "32ed17ecaf7548d5a80b27eb97cd78af", "56817cd3b11e46f187dc13e71b7ec97d", "3d30c20af23e495999646521d39e8e66", "bafada4a1f29442296c47018dcaedb77", "6cd3baac550741ba866ce2cd2dcaebdf", "bc9c3ba4bc6d4623a764d92890536935", "719f2b2ab9eb4e348ae902d063c27e2d", "5e09bdca5081429ab82b488a3e016fbf", "200bc7d8c2bc494399a4560dabd64293", "91c3b5ddf5fb4c7aa835d992b4c7b4e7", "52ff6528abdd4a9e9b85f4b355b2391d", "6bb6737e22bd48d786b02051d077e8cc", "e97a0201b81446b882ba802a5a3b00e8", "c1a89a042b044278a8666da306e6a481", "21ce216193f346c09a04a1d64bc0cd8a", "23ee9c5c18c64305adc55ee218afd7a9", "1f632bdc00e84422b64c0a4b8e238f34", "8d50d82f9d94482c9883f99ea5c7d704", "6a2dea21e7ce4eda8953995497308327", "2f3d1d2f1c92402cadc4cfa1b0094238", "844caba5ebdd48189517a10705543292", "2d667cd4bc134a44958d17ef75d86321", "acfb7b6734884939a753fc19047bb9bc", "73cd51c6c43b40cab25d411e7c0f6ad1", "19c7ea1e9364437a9c0bf6b4e645f854", "8675aad817a44ab39b097ec864669833", "5664e1233a904f4bb4af9c5322517fab", "0ae5110b687440e89ebebfb847985aa1", "85509b0aafec47e5ad540abc7ae4ab7d", "d3d7c15d53c8498e823c84fe609321dd", "80de3739b91a45478cb6a00dcaeae756", "7c99e16722cf4fab866732557769b921", "00834c084ddb4f46ac29f14575a2383d", "4ce1e695a6ab4906ba0817005cf58d55", "eb03476353ef4568b94a3071918e72f2", "1b8307d133bf4280845f7d3a302b3a2b", "daee15869a92459aabcd9128526183d5", "5ddd9e1a1fab4531930acaf08cc45c73", "4081e141986f4abba15477a12a752ca0", "892e150a80464f8198024587a47186ba", "9f9e15ff2e394ee7a7776e3e7f4b7d30", "7e264d54d38e4d02acfd47e4e533b49d", "746302ee5d21495db5d9a13789c5256c", "47f7ff7b213e405a822ad6fe2e5de431", "e4810a798c0f47b6b54f84ff4ffec608", "8e480fe546f24afebb1ea723bff84456", "0cf925f582374cd197164625f0ddf27d", "aa7557fbffc54ed3a9c58c1531fd93f6", "061c90e4148b43b8ba65848ca4c1ba46", "f617d181ffdb4b1e8d81fbb393923a6e", "a26ddb684a07496da4290e3f6031b685", "f20fc82bcbf245619ad4dec04d0f999d", "8016eabfc7aa48c2bc0bf3a13919a675", "83cb83e404a24fba8cf43610cb1696fc", "7b70db203ee644fe81006f5d48aa426d", "efd9d5724dbd435991052b4445c6970f", "41f65340441f419d91e1d3841921f48a", "58aba8edefa44c60818891a2651137be", "e5e8f119a91944f296ff821dd7ecfb1b", "568a85caf800461b8571e3d206854e6b", "e2a8a379bd0d4cbdb22fbc3bbb4fdc7a", "2c7d952f958247b681301d1c6bff4fa7", "27649a0d303643d8985caf506ef66fc3", "1855ae0388074d08a4763b7ac76c6948", "7e5252f608d6468f80deb468cb2556fd", "bce6b62300d942a1ab89a6f0ceb16d30", "0239f7263d1a4e8b9475ae48379c068d", "5b527e45d6a946b28047115fa6b5aa3a", "3c5eb07f0bff43948ff1f283c3c56b0f", "3545fcc1e9d7453099c4931f658e0bc8", "2f314258091a430095794c3fe0aea7e3", "0ebfa123462342e99ad81b7e5ee00eac", "8af6bd6e8cae4964a2e372be30220bd9", "4184f160ab6d4f58bc921fb7ba89cf60", "575c4d8d503244d8a9f2a5709021b5b2", "ffce35af1ad84a2c838cca55a26dd3c4", "d21cd4878c6e49d38dd3abb2e3b3f566", "969e3cf7f3634c3f90b5fea38c5797ca", "78501c2a5ac84f9ca0d20bbef340fc9f", "5b4fbd1102a84670a1eed6f3d25c0bcd", "621880c98245427881dd5b004b480c6a", "65c2716dd7f14afa93d3bfaebe85c44f", "2ff541d18f5844408690be00c7259d59", "679fa9d2a703494dbb10fbfd19879f1a", "6717b182e5674afb90006b50dea98cae", "0887c7aabbbe4d4e8fdc64d2e2657cc7"]} id="a2787582-554f-44ce-9f38-4180a5ed6b44" outputId="b1729004-591e-41c2-c206-9d0e572eb8e5"
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)

print(common_voice)

# %% colab={"base_uri": "https://localhost:8080/"} id="20ba635d-518c-47ac-97ee-3cad25f1e0ce" outputId="dd81bced-f544-4d55-9669-5babb901f842"
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)

print(common_voice)

# %% [markdown] id="2d63b2d2-f68a-4d74-b7f1-5127f6d16605"
# ## Prepare Feature Extractor, Tokenizer and Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["4d25d9919acf44a19f1b6f8fd625f808", "b041efb27b6149fcaf206590f1c1b961", "4668602d021a400a8b0ec88599abc85c", "92d9732acb964601b27695041f9fbb72", "c3ea17f7dd94462986d30b751664d77b", "1b1dc31d9a2b4357a71119300c6d899a", "7a24ae9d13fb4e82b1993b05f8d71d11", "e1141861d9f44d4c95313fd432795b70", "6ef55b5db76d4c78854fa0c27165e480", "91b103ac79e641b190416acdeb55902c", "e9e10f1e53b74509bfc9c0bf11502c5e"]} id="bc77d7bb-f9e2-47f5-b663-30f7a4321ce5" outputId="7abb2062-e755-4f1a-e88b-9b7bf2986dbf"
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

# %% colab={"base_uri": "https://localhost:8080/", "height": 209, "referenced_widgets": ["f436e6a7d3014e2ca44c94455bfeace8", "f9d6d41ffeba43cf94ec9d7a96af6617", "bed65245e7234977874d35bf78694e35", "c28c2fc81e0e467cbc0ded0205d1ee85", "ea941a078b984f51b66b5f1e8f3d1d82", "e3f1244dbe2c48bc8102f958c3df6467", "786d3d34cfd24f81996797c55b10b443", "3a4bb5b9cf864265af8f1a0b989dff2c", "ac2e1977ddaf4b948c9cc24d84c08b83", "dd339bf6baf6433e92665e30dc30b062", "6a544d6dd5954beab32ce3089d8d2aac", "f933fbdcc26d41b0bb294dabd0337834", "950e74921e6042868ab6b7b9070d6f69", "82dc91c5b065459d827863699a9710e1", "5b7b6f08765c4c1989f945414f2c3cf4", "735c3606df924b9297ddc05fae3e92d5", "d5d632dd16f147e090c62aad38c45d3c", "24191f38e3654b86a5da3576615e2229", "4b1f4abe697948d5a1cca9b45b2a6f87", "8527f474f1a549abafe9519e2b4bd338", "1031dc4b5d2d45f3b14ab12dbd9bca56", "ab40616ed9b74f438e74d403f848bed4", "e6c2dc814c324a0c8cb744ca16707479", "a615446a48624d0f9a009c1f6d8b1a54", "383d6e891c5249b4b2fabb60d4900488", "3c3214f235a54864848902f7e53662db", "73f6e6860b64491284c9442f0deae8b6", "6d9826a5adb847e8b4cc4486a31b52ce", "e8b3093587e44164b0ac043414cea0fa", "96266c5722f54eeeb682b2707b8025dd", "f0be69583cd1410da6dbc18302d4439b", "c24ddbaafa5f4a63b391d981a4f10354", "629725dfcc684b1db1a57850c3bc7bc3", "add0f175631742d49dd4695bc004e81a", "5871a46d60f14da99e4bb8ee74405319", "64b308019fff4095ab7f1812aab4676a", "03f62b3f5aa64b6390b157cb3f9d2b9f", "59657f26f4af490980b1d9bea1526e5e", "1571fa5d8f494704949c5809f3409fa7", "5ef218872f2646249888def0abed7149", "03fcbc61c50c4ddc9c3684e4c085bffb", "e4234c7c29744fc4be99b8b2ebedc9d1", "70aa1fc14b09473e911dad3f9030b15b", "008bead021ea416eb31dd9143d5ae5b9", "ebd79e22ad4e4256a6d88883af2c1eef", "c47ba0b11e074f708338863a35a78f7b", "72b715be2c774235a21602b18d71e75a", "ff9a0ed54bab49aca6f27bf1be66958e", "15b5e415b62146ba96215458cf116431", "8b0dd001d5b04647b1c480aab83d03a2", "0af935ad4f694fb48094e6a119cbcf82", "7cfbb542eba34459ba64b881c1040eee", "fe48e65b2371445bbb01a8d3e9af1f67", "b8d3539c8a454217a3b6ffab51259054", "05e96819517e417aaf05f5f38c0c8b76", "b600ead93bbf44a3a3fe229589c63a61", "3d4c614fb775434fb0c5b02d46246ee0", "344d1cc53e28411cae839a4ebba1bf58", "a1578c0c777f4780a3fdd1635a0909d9", "41faf55c8878475f8a986b02ad73e8ce", "b5c8221a09df4dfdb74017d4af544b95", "65e4b8dae1e4435cb7737a8d5dd13d91", "6d932eadfd6a448a9a71d741bb64f428", "dce0d285c7d947dfba9ee5bc1a6ebece", "bde010029c374a0eb2bb942f380f0e8b", "a91f37ce798d411ea2a33cfdb1f01251"]} id="c7b07f9b-ae0e-4f89-98f0-0c50d432eab6" outputId="4c094075-3f95-42db-9f4b-4db0c2fbeb91"
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

# %% id="77d9f0c5-8607-4642-a8ac-c3ab2e223ea6"
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# %% [markdown] id="381acd09-0b0f-4d04-9eb3-f028ac0e5f2c"
# ### Prepare Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 72} id="6e6b0ec5-0c94-4e2c-ae24-c791be1b2255" outputId="1f1fe2d1-3ad2-42d4-e6f0-f0929785ae8e"
print(common_voice["train"][0])

# %% [markdown] id="5a679f05-063d-41b3-9b58-4fc9c6ccf4fd"
# Since 
# our input audio is sampled at 48kHz, we need to _downsample_ it to 
# 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. 
#
# We'll set the audio inputs to the correct sampling rate using dataset's 
# [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
# method. This operation does not change the audio in-place, 
# but rather signals to `datasets` to resample audio samples _on the fly_ the 
# first time that they are loaded:

# %% id="f12e2e57-156f-417b-8cfb-69221cc198e8"
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# %% [markdown] id="00382a3e-abec-4cdd-a54c-d1aaa3ea4707"
# Re-loading the first audio sample in the Common Voice dataset will resample 
# it to the desired sampling rate:

# %% colab={"base_uri": "https://localhost:8080/"} id="87122d71-289a-466a-afcf-fa354b18946b" outputId="727a709a-2b21-4c54-807f-efd40ea1719c"
print(common_voice["train"][0])


# %% [markdown] id="91edc72d-08f8-4f01-899d-74e65ce441fc"
# Now we can write a function to prepare our data ready for the model:
# 1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
# 2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
# 3. We encode the transcriptions to label ids through the use of the tokenizer.

# %% id="6525c478-8962-4394-a1c4-103c54cce170"
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# %% [markdown] id="70b319fb-2439-4ef6-a70d-a47bf41c4a13"
# We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially.

# %% colab={"base_uri": "https://localhost:8080/", "height": 197, "referenced_widgets": ["466eeda389c442e487742faff05eeb81", "664ff40a4e7346869652ba3be663acf2", "3d8f10b5726b46de934dd0f7ed7244e4", "b803893882ac4358a4676964dfb3fb31", "4ff7e3d07f6a48f8935f0f42b86fc20c", "3e5db1000a6c4871936614e712e769db", "21dcf39a421847a78ea685eec0983fc6", "c4119f7ec7464aab90f17d022e674999", "965ac01b68064dc5a9fc3bb3f244c804", "f5d7433d15de45e997d12568cac536fc", "14e284f308844311a9ad40415091d93f", "58c3766293e64116a131357f6cc66fe5", "abc8f69eae7b46b1b430cf3b7a231b05", "979edd227f0840a4be0233082e452b5a", "12dedd9f089c42f3a8bba9adb01cf9d3", "3042d874fcf544fabc4701938bca7cf3", "baa8f739bb7e402bae7ee479e282e813", "54509d703b7d4416bdee59157f918396", "ec6017ce2fb3431ab823bde05f977a61", "1a540a7cba794122abdb1900061479e2", "d046a46c70ea46ffbb04a3c9f55637d5", "99b628071c814d88a3cd5d72e4c95f01", "e213d1c919314315ada180d49e27dfb6", "0ea1f163e4174684bd6efc2e2433c1d3", "09511a81a89d4754897a3507a84405be", "8338339ab8a242c1a485bce8558f9c39", "8e84abf61e3e45d58efc7ccd0bfe8d37", "47a5e0cfb3564c16acaedb88613b9020", "09e357a7187044b0bd2b841893a2601f", "65e8c45d05be4d41b136439d9785a09c", "1442a54fad7d44ddbafdacaf7f95279a", "4bea6b0455ae4019b33d45491dbf4584", "9c928a14371a4d9aa521953e287fff54", "b8acdb71c3564972a6c8b27e964ec061", "688f552e85714fe6a5d3eda82be0106a", "a8aac44a077040bd9f4638c0c8f7a877", "89f8ede64593475da4e056e9907f370f", "851902cb2b0e494998684409d82dafd1", "328f4d11886d48f49d4578e8e352097f", "a25ef41450eb42ff9b8b618b40080ac0", "04144ed3ff5f423f99179702cee5343f", "8c41e1ee1a2c49b8b3d3d320fbf26262", "4edf0948ed1346ceba1c0148e9c6fe1c", "40f2b2ff75ca4562808fc2c7a870901e"]} id="7b73ab39-ffaf-4b9e-86e5-782963c6134b" outputId="eecac4c8-c5f8-427b-a2ba-0d51fae825ac"
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

# %% colab={"base_uri": "https://localhost:8080/"} id="c4be572c" outputId="0383124a-d4b1-4abe-a8a1-868ce6b3884e"
common_voice["train"]

# %% [markdown] id="263a5a58-0239-4a25-b0df-c625fc9c5810"
# ## Training and Evaluation

# %% [markdown] id="8d230e6d-624c-400a-bbf5-fa660881df25"
# ### Define a Data Collator

# %% id="8326221e-ec13-4731-bb4e-51e5fc1486c5"
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# %% [markdown] id="3cae7dbf-8a50-456e-a3a8-7fd005390f86"
# Let's initialise the data collator we've just defined:

# %% id="fc834702-c0d3-4a96-b101-7b87be32bf42"
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %% [markdown] id="d62bb2ab-750a-45e7-82e9-61d6f4805698"
# ### Evaluation Metrics

# %% [markdown] id="66fee1a7-a44c-461e-b047-c3917221572e"
# We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing 
# ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["215c3486e13343e091a26d658e1030d2", "999382fb9e764a98893bd5269261d70b", "4e9729771294424f959bbfb6bf8b60f3", "3d2249516ecd43a08b5fba53fddb32e8", "681711ebd0c64a63bee6b5337ef401db", "d37791ea2b6c4152991295ca0edb0fb7", "730f4d93bbd94452a59de43bf3d0a266", "048e5c87cea34014a8f8a4538f4126bd", "fbd27061ff114846aedc99bc2d17f7a7", "5a306409e9e045b2b936267c520f935c", "9f0638198f544a3bbb31a3a78b7bd2c2"]} id="b22b4011-f31f-4b57-b684-c52332f92890" outputId="b0a08086-69b9-4ab4-97ac-dbed295f2e15"
import evaluate

metric = evaluate.load("wer")


# %% [markdown] id="4f32cab6-31f0-4cb9-af4c-40ba0f5fc508"
# We then simply have to define a function that takes our model 
# predictions and returns the WER metric. This function, called
# `compute_metrics`, first replaces `-100` with the `pad_token_id`
# in the `label_ids` (undoing the step we applied in the 
# data collator to ignore padded tokens correctly in the loss).
# It then decodes the predicted and label ids to strings. Finally,
# it computes the WER between the predictions and reference labels:

# %% id="23959a70-22d0-4ffe-9fa1-72b61e75bb52"
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# %% [markdown] id="daf2a825-6d9f-4a23-b145-c37c0039075b"
# ### Load a Pre-Trained Checkpoint

# %% [markdown] id="437a97fa-4864-476b-8abc-f28b8166cfa5"
# Now let's load the pre-trained Whisper `small` checkpoint. Again, this 
# is trivial through use of 🤗 Transformers!

# %% colab={"base_uri": "https://localhost:8080/", "height": 220, "referenced_widgets": ["d8c1a66480204f1095ff5f6a7dd2e477", "9dc736113ef6477d91aaf71c9969ca74", "79521628c64b4d6f9f22e73749298693", "c7268972f75e4824893ebe7d893a18e1", "a87598d464174703b5f5a5eca23543f3", "e0529b81739144db8912c2d6789e729a", "13d0a97497274652b081cbcefb3fd17d", "08b26adf061b48f59078bf0c0b59e643", "ef056ad59e314089a012acaa73a54e4f", "7935e298049f4deeaeb278ba3de92291", "3afca90970fc4925a05a9a1aa5c8d2f2", "40aba44ef0a74c0d9a385c803c1365a6", "ae54388d78dc4b7ebd1b21860421ffe4", "bb4a47c63d254d4aa3220e85f86d37c4", "cd585c98560b42c8b4a08df5b853b23c", "a140cd385e5a4c0a88c5562370982d2e", "fce64f5690024c698701330f0e5d039a", "abd0e7c414974e51b280a29bf978f776", "bf7961a79c2f403a89a4fb6d4b1a02e5", "1d8636d1d1c3442fbbc2fc83cd03fa44", "313090dc1f034ab19d5ffc573ea1aa5c", "966c0400dafe4e3ea2c9baebc2e104fa", "9087be5d992c4198b3ec2c61f4021164", "6bc07db3471342a6bbc1b86ca734b3e6", "4f167e9657274d56b8568d48262d1ee6", "b7fbaa9d4bcd40b5bcf8d1475658c5b1", "ba21f5ddf2434cc792b70a70c8c1079e", "21bc84c704874455a57c47239984d28d", "5b226e580df94448bed54b400c7a9a25", "b0237154051343adaa076a9cc6dd711d", "64756452791144859b9c803886c6dc77", "f027b4358c2c41a8a93c617641135bcd", "42cb6114956c4a86980c8dafd5a734ae"]} id="5a10cc4b-07ec-4ebd-ac1d-7c601023594f" outputId="163d4b39-7e5d-4126-8d78-846c5d94dca6"
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

# model.hf_device_map - this should be {" ": 0}

# %% [markdown] id="a15ead5f-2277-4a39-937b-585c2497b2df"
# Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):

# %% id="62038ba3-88ed-4fce-84db-338f50dcd04f"
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# %% [markdown] id="bR-_yaEOPsfQ"
# ### Post-processing on the model
#
# Finally, we need to apply some post-processing on the 8-bit model to enable training, let's freeze all our layers, and cast all non `int8` layers in `float32` for stability.

# %% id="Cl_ZQualPt9R"
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# %% [markdown] id="Vjl4j4RJPmPR"
# ### Apply LoRA
#
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

# %% colab={"base_uri": "https://localhost:8080/"} id="DQtpDPRHPyOL" outputId="1effcbde-7acc-4f62-f24b-e6236a43f833"
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

# %% [markdown]
# We are ONLY using **1%** of the total trainable parameters, thereby performing **Parameter-Efficient Fine-Tuning**

# %% [markdown] id="2178dea4-80ca-47b6-b6ea-ba1915c90c06"
# ### Define the Training Configuration

# %% [markdown] id="c21af1e9-0188-4134-ac82-defc7bdcc436"
# In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

# %% id="0ae3e9af-97b7-4aa0-ae85-20b23b5bcb3a"
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="temp",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    eval_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

# %% [markdown] id="b3a944d8-3112-4552-82a0-be25988b3857"
# **Few Important Notes:**
# 1. `remove_unused_columns=False` and `label_names=["labels"]` are required as the PeftModel's forward doesn't have the signature of the base model's forward.
#
# 2. INT8 training required autocasting. `predict_with_generate` can't be passed to Trainer because it internally calls transformer's `generate` without autocasting leading to errors. 
#
# 3. Because of point 2, `compute_metrics` shouldn't be passed to `Seq2SeqTrainer` as seen below. (commented out)

# %% colab={"base_uri": "https://localhost:8080/"} id="d546d7fe-0543-479a-b708-2ebabec19493" outputId="e2fabe64-2c50-42ff-a7ca-7773813e9408"
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ee8b7b8e-1c9a-4d77-9137-1778a629e6de" outputId="cdea5268-f33a-4d48-ea4a-a9c71576f81d"
trainer.train()

# %% colab={"base_uri": "https://localhost:8080/", "height": 116, "referenced_widgets": ["f309d7a096df4f119e6e6871b56913f1", "5f283548f34848af90affe55a169b5a9", "51a4d85c08d745bda70ba0db731dca68", "eb46a602b3ef484daddbcb847957c3e5", "8e2e49c6046e4dc0a0a810b4e58f80cc", "f3f191968f724e9bbb710b86d3657a66", "d97f77f45e1e400494c2fbf2cd9d69a3", "10e52425e59d4c2c8d2a1239b24e5a95", "9ca0667998cb443d9df29fdd09cf90ff", "f22aa0a93d8b44d0a4c412343ec1f48b", "6e6a59f8e7454c2d886634eade47a21f", "92ae14b97e814d51b016e2a14f227a15", "fb8aca596a1c4ac4a7428248b6c6f8b1", "31de6bfcde21400ab02af4fb31d409da", "3c45a19fdf664b92a27be16abf537a03", "9827e382bf3b49b092969d5656dcad7a", "168e19229aa5404bb151e4547bb31283", "d140f5373bf144d1ae5d282e1a65647e", "fa137c931d2c4e579c27893ca8ee1848", "93647f2d98a74109bfc5ca7a9cc23eed", "a478b23c34654615aad4202b8c7089f6", "4553ba4c0ed645dbb9ddfcc88975ab5f"]} id="0576aa2a" outputId="6d3fe4aa-f42a-4428-e070-2272f3b87f5c"
model_name_or_path = "openai/whisper-large-v2"
peft_model_id = "smangrul/" + f"{model_name_or_path}-{model.peft_config.peft_type}-colab".replace("/", "-")
model.push_to_hub(peft_model_id)
print(peft_model_id)

# %% [markdown] id="SlyyOGnPgi_I"
# # Evaluation and Inference

# %% [markdown] id="Kzfg2qoXgrhg"
# **Important points to note while inferencing**:
# 1. As `predict_with_generate` can't be used, we will write the eval loop with `torch.cuda.amp.autocast()` as shown below. 
# 2. As the base model is frozen, PEFT model sometimes fails ot recognise the language while decoding.Hence, we force the starting tokens to mention the language we are transcribing. This is done via `forced_decoder_ids = processor.get_decoder_prompt_ids(language="Marathi", task="transcribe")` and passing that to the `model.generate` call.
# 3. Please note that [AutoEvaluate Leaderboard](https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=mr&split=test&metric=wer) for `mr` language on `common_voice_11_0` has a bug wherein openai's `BasicTextNormalizer` normalizer is used while evaluation leading to degerated output text, an example is shown below:
# ```
# without normalizer: 'स्विच्चान नरुवित्तीची पद्दत मोठ्या प्रमाणात आमलात आणल्या बसोन या दुपन्याने अनेक राथ प्रवेश केला आहे.'
# with normalizer: 'स व च च न नर व त त च पद दत म ठ य प रम ण त आमल त आणल य बस न य द पन य न अन क र थ प रव श क ल आह'
# ```
# Post fixing this bug, we report the 2 metrics for the top model of the leaderboard and the PEFT model:
# 1. `wer`: `wer` without using the `BasicTextNormalizer` as it doesn't cater to most indic languages. This is want we consider as true performance metric.
# 2. `normalized_wer`: `wer` using the `BasicTextNormalizer` to be comparable to the leaderboard metrics.
# Below are the results:
#
# | Model          | DrishtiSharma/whisper-large-v2-marathi | smangrul/openai-whisper-large-v2-LORA-colab |
# |----------------|----------------------------------------|---------------------------------------------|
# | wer            | 35.6457                                | 36.1356                                     |
# | normalized_wer | 13.6440                                | 14.0165                                     |
#
# We see that PEFT model's performance is comparable to the fully fine-tuned model on the top of the leaderboard. At the same time, we are able to train the large model in Colab notebook with limited GPU memory and the added advantage of resulting checkpoint being jsut `63` MB.
#
#

# %% id="273a996c"
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

peft_model_id = "smangrul/openai-whisper-large-v2-LORA-colab"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)

# %% id="401ceaa6"
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute()
print(f"{wer=}")

# %% [markdown] id="8ZN4pTangw98"
# ## Using AutomaticSpeechRecognitionPipeline

# %% [markdown] id="hpDx1AOCgwuk"
# **Few important notes:**
# 1. `pipe()` should be in the autocast context manager `with torch.cuda.amp.autocast():`
# 2. `forced_decoder_ids` specifying the `language` being transcribed should be provided in `generate_kwargs` dict.
# 3. You will get warning along the below lines which is **safe to ignore**.
# ```
# The model 'PeftModel' is not supported for . Supported models are ['SpeechEncoderDecoderModel', 'Speech2TextForConditionalGeneration', 'SpeechT5ForSpeechToText', 'WhisperForConditionalGeneration', 'Data2VecAudioForCTC', 'HubertForCTC', 'MCTCTForCTC', 'SEWForCTC', 'SEWDForCTC', 'UniSpeechForCTC', 'UniSpeechSatForCTC', 'Wav2Vec2ForCTC', 'Wav2Vec2ConformerForCTC', 'WavLMForCTC'].
#
# ```

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["63d6ac9358774b269989091ca090e369", "eb0d8c3f6de3468fb2b41ddf940de999", "8c010800d6d945fe965fd5b9c6ab2ec4", "0acd5b5ce27b4645ac9c598796fa0648", "7706a602e7b64621a81777aa3229c161", "3547b43905c94f479248a29b60e5a1ce", "02ec476d0571401eb18332d2af3a13fd", "4568923487c44041ab985ee326842d15", "352a2237356846af836779413e584fee", "c80954848a4a499a87708548bed8e7cc", "0044d766693a47e7a2160044bf916ecd", "f96c7014b47d4d048ff50d6f1f2da200", "925f7aad24b547cc8071ee9bda713d7c", "4043e3ceb27c435d83913b3796bd9927", "b1183b9042744d3fa7437e2ec55b6cdd", "20a0694ee1684ffc8289af9094e812e1", "7e2534cfd8564dbd9bddaa2217f2dda4", "a5bd56eb69524729afdfdd14b55b6130", "d2c14b2486e24150a8242ca37f4200f4", "52c3c792386c4db6a1fcd5d2bd88415b", "c643b8650c31466a8614510928ab5d2e", "58e46e1d355244bcab90b32c163e2877", "568811a89e2a474fafe9d66bdfb4ff49", "dbe19b9505884a958e832c0362d547df", "4be782f67d084a6785f163126f4ef829", "70222321ce9c4932b7c74055b4f773f9", "c4e7cf4e42554d61aa055da47ab0ee22", "6a40bc08e55a4605b478edf01ff088df", "fbd304c2564e4acba84809377ba19e25", "5a177ec321954dca9ae74ec30f39e2a3", "3e0ce97eb4694a689316042083c0a8ef", "f5ff1816a56243e4872ddfcd35331ad8", "4a86fbd37d224ac488fd2c53ab02724e", "74684eea520848cab51a1428823d5008", "14c618afb2b6418a8b8782b102004e4c", "d65e79ed6a9f4daea04e098959f12c80", "71805c4082944d339a2128f7523d3f89", "bb1bcdfd39bb430c8a453d7a4c3a5e3d", "980e87bb43d54306aba5019964857b89", "2c2b50f5cd7d4da0ac248e0ccbf13797", "c89cb73d51dd457d8c17ff97e74b7ca1", "921fd273a7254447a93fb997773aedab", "240911ebfad1435ebd39d8942509a2b9", "635164697bf74a25a8ac1af286441e3d", "0e68ff83e8e64597871aa90cc871e15a", "ed80930617a64700a92b3bcff97c2885", "be90fac89e5243fb82389ed06d094c10", "92debeaea5cc4136bb37703947e825e0", "96c0b77f9a5e473b8b5dc92bde0f4a9c", "0dc05bee870740ffa72af34ace8c05d9", "7a8b8f2a2160441dbdc54a62de83d297", "845211300fbd49f480d056bba9de83e1", "df4435b0da414c3880adf33a50c465e5", "7aca78c9768a48a7ab0a8220a3b364b9", "7213f79093f74df996f8dbe7fe816ea0", "7ace5b5107ac4e8bbdf4e3f81b73bc07", "2bad57f7004f4f8fbd7e3b6c7b1d339e", "5226b933249f4a48848de7e32692f463", "830245f63f1947e18b29986d9a091c0a", "1f995a2a5e48449b90e04c2ca9cfda18", "e6dcbcdf41c34e418be0b0b86e44d3f1", "812f78ca88234efa9b2ccc001245290c", "bc2b317065ed4b658995663ff202108a", "251c13fd793c4a6d934c14cb94f57436", "b0efdb97594b4997a8cb148aa03f9a6b", "66bdd68e31204abc92fafc0e29860d1c", "bbc5dbbb8b3548d485b878e194683cf2", "c0432026ca204c4e89721fb22589c576", "c5672e82461148e29dc93af865e259ff", "f8053b94954d401a85ead90319aa4690", "cfee6cbc1d15435bad0bc4a193542c10", "35e6c9e3c5ea47a89cd0cbebfd197bde", "f81bffac74ff4b07a09760a00620610c", "cb55e8ca8bf4477bb323ae45a6c60500", "3e154f89e56146438bc45030e5254d8d", "9abe7ed3e3d347b6bad7d1252bde226f", "8ba244a082284f6dbac1c8f689c527a8", "fab643980982452988a02c14132c73ca", "062727d821f248c9958ecc2d79237e5b", "2c264564d4c642ebb033703b1df02c69", "b19431ca425341568c2b3a8556431a8f", "441e89acb62e47dab61b32b5a97110f3", "f93f87ef211446379963df5bbb2e4ff0", "7fbb2599863f45f49ad4925d75ef5f77", "06af43189ed6478b8dce4bc1cc31e657", "86692318e7584e62b4143c1b2eca5935", "6072c153de2540bf9e1227db5057fffc", "b7d9e1bf1d2e48e59c4dca38a638d4b4", "f09d5d5e7faf48f5adbaa90cbcf55162", "3ff0c6a4f4fc4b72a5163a7da50af4da", "e1fef249d35d482681cd39eefb7e0d7e", "0a8ca977f9db4d7794dcbcaca3c5cb96", "e7be6d842a0c4ba485dfa8e58338eed4", "04d485813a344bf5922a2f3891f0b6e8", "c78544f32b564ba6832120b3167a2162", "54af0365067b498d86d23c453c5e38f1", "ab4ae16fa4f448838f677ea60523c905", "d92259acf9704ef9be6eced8a5f25dab", "ef01f9efbfb749a3999c5b89ba0ee370", "78d28e020ff048a1aa05818ea778056a", "204e994a8270488a97ec2114d8ac01f6", "b3a59b83acaa4864be23e10d90560139", "106c4e8e31f3406f9165717c8e92003e", "36ddd7c4059a4b49ab7036cb9d945b43", "3b80c40ea02149989fc7042362ec452c", "590ffe12145e435f97b7dcff5ebf5bcd", "7b8266519f0a41cd9a065b9e795d7e84", "658361b3d5054f44a5c21df747537bdd", "83a7cd2ce4074720acc6236aaf24014d", "50cf6847ada943729e253ca09ee7832c", "1f0004d38afb4635978a8ff5a654e67d", "04894f85598647b08405fae8d4ca0421", "8b9324f5c0e24576a9733f30a8945edd", "e166fb45a33c495e9428bcdef8cd8813", "05f032a5fed940bd9e08425cf15b1aea", "35aab9e31bbd43dfbc64008545598fad", "7847c294b90242c2bf7feb156b1c6b36", "ad8b6c32caa4493ebadf7691ed1546d4", "85cdb3ecd88d426a8e517b3c8c8a8307", "098e92acdc444f64a031bd0daf070604", "6feb1c05469b4bc8ac9c98cdf29fdd57", "f9e5029c15054f5a9cbcda1ed5878995", "4cc5b0a5c84e43f3bb185c63bc1f9a56", "af56501692b84e718fe3e8b1e452b7c2", "c6c9fa535ff4458ba966183e1489905b", "b0e084a4c551427bb176061df894fdf6", "243bb4a421974846bab104aec541d955", "42fc47a3348f4488a03665a01af273cb", "e53d35abac8241bcaf6d3b0f732d1bb9", "87d251db4a6e495cb1d59d276e823bf2", "56e8812fa66e40b0896a19dd3be7330c", "116c89bb3a2045f28123e4f36067a7e5"]} id="fa2b62a3" outputId="d000c2d3-af8b-4a0d-ce50-6aedb543463b"
import torch
import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig


peft_model_id = "smangrul/openai-whisper-large-v2-LORA-colab"
language = "Marathi"
task = "transcribe"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
)

model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


def transcribe(audio):
    with torch.cuda.amp.autocast():
        text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
    return text


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="PEFT LoRA + INT8 Whisper Large V2 Marathi",
    description="Realtime demo for Marathi speech recognition using `PEFT-LoRA+INT8` fine-tuned Whisper Large V2 model.",
)

iface.launch(share=True)

# %% id="XNn49f3BiLXa"
