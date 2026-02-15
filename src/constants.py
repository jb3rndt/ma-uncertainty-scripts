from pathlib import Path

ORIGINAL_DATA_PATH = Path("../data-pollution/data/original").absolute()
CLEANED_DATA_PATH = Path("../data-pollution/data/cleaned").absolute()

PERSON_NAME_REGEX = r"[\w\-\.\&']+(\s[\w\-\.\&']+)*"
PERSON_LIST_REGEX = rf"^{PERSON_NAME_REGEX}(,{PERSON_NAME_REGEX})*$"
# fmt: off
ALLOWED_GENRES = {"Action", "History", "Fantasy", "Adult", "Biography", "Comedy", "Musical", "Romance", "Sport", "Drama", "News", "Family", "Sci-Fi", "Western", "War", "Documentary", "Film-Noir", "Mystery", "Adventure", "Music", "Thriller", "Short", "Crime", "Horror", "Animation"} # Acquired by listing all unique genres in the dataset
ALLOWED_LANGUAGES = {'German', 'Algonquin', 'Assyrian Neo-Aramaic', 'Swahili', 'Somali', 'Hawaiian', 'Inuktitut', 'Thai', 'Pushto', 'Korean', 'Pawnee', 'Lao', 'Flemish', 'Maltese', 'Icelandic', 'Syriac', 'Esperanto', 'Greenlandic', 'French', 'Zulu', 'North American Indian', 'French Sign Language', 'Scots', 'Quenya', 'Welsh', 'Croatian', 'Latin', 'Cantonese', 'Acholi', 'Tagalog', 'Norwegian', 'Nyanja', 'Navajo', 'Creek', 'Egyptian (Ancient)', 'Filipino', 'Tok Pisin', 'Breton', 'Hindi', 'Bosnian', 'Turkish', 'Ibo', 'Khmer', 'Romany', 'Bengali', 'Aramaic', 'Amharic', 'Panjabi', 'Southern Sotho', 'Old', 'Hmong', 'Mende', 'Ukrainian', 'Serbian', 'Tamil', 'English', 'Luxembourgish', 'Danish', 'Xhosa', 'Albanian', 'Sicilian', 'Japanese Sign Language', 'Latvian', 'Polish', 'Greek', 'Vietnamese', 'Estonian', 'Scottish Gaelic', 'Nepali', 'Malay', 'Berber languages', 'Tibetan', 'Norse', 'Spanish Sign Language', 'Sign Languages', 'Armenian', 'Hebrew', 'Slovak', 'Polynesian', 'Gujarati', 'Mohawk', 'Burmese', 'Aboriginal', 'Georgian', 'Yiddish', 'Cree', 'Sanskrit', 'Hungarian', 'Swiss German', 'Old English', 'Klingon', 'Chechen', 'Romanian', 'Irish', 'Basque', 'Dutch', 'Tonga', 'Brazilian Sign Language', 'Ancient (to 1453)', 'Urdu', 'Arabic', 'Belarusian', 'Ewe', 'Maya', 'Spanish', 'Dari', 'Kinyarwanda', 'Indonesian', 'Japanese', 'Swedish', 'Italian', 'Mandarin', 'Bulgarian', 'Czech', 'Portuguese', 'Serbo-Croatian', 'Cornish', 'Afrikaans', 'American Sign Language', 'Sioux', 'Hopi', 'Persian', 'Sindarin', 'Finnish', 'Apache languages', 'Sinhalese', 'Maori', 'Creole', 'Crow', 'Chinese', 'Shanghainese', 'Russian'} # Acquired by listing all unique languages in the dataset
# fmt: on

OL_COLUMN_CHANGE_RATES = {
    "author": {
        "avg_changes": 2.8366738154517856e-05,
        "avg_time": 3238.2092169591638,
        "null_rate": 99.99878974497909,
    },
    "author_names": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.9999394252547,
    },
    "authors": {
        "avg_changes": 0.08336869484758914,
        "avg_time": 2541.418977296649,
        "null_rate": 12.967036356719841,
    },
    "bookweight_unit": {
        "avg_changes": 1.7397355601948505e-06,
        "avg_time": 5416.584752099259,
        "null_rate": 99.9996365515282,
    },
    "by_statement": {
        "avg_changes": 0.00237272189207989,
        "avg_time": 4138.006353709576,
        "null_rate": 54.89252544274565,
    },
    "by_statements": {"avg_changes": 0.0, "avg_time": None, "null_rate": 100.0},
    "collections": {
        "avg_changes": 5.56881795076628e-06,
        "avg_time": 3563.4601605657053,
        "null_rate": 99.99933443652361,
    },
    "contributions": {
        "avg_changes": 0.0006383296636657999,
        "avg_time": 4269.553478418656,
        "null_rate": 76.92057058864341,
    },
    "contributors": {
        "avg_changes": 0.00312334110458999,
        "avg_time": 1632.672308778373,
        "null_rate": 99.75750717254662,
    },
    "copyright_date": {
        "avg_changes": 0.00303472550135798,
        "avg_time": 1542.3744165388744,
        "null_rate": 99.49114962119211,
    },
    "coverimage": {
        "avg_changes": 5.864378031440131e-05,
        "avg_time": 4408.216621217835,
        "null_rate": 99.98626463189127,
    },
    "covers": {
        "avg_changes": 0.1712877552756112,
        "avg_time": 3900.253777412511,
        "null_rate": 72.12881197606005,
    },
    "create": {"avg_changes": 0.0, "avg_time": None, "null_rate": 99.9982150749281},
    "created": {
        "avg_changes": 0.26271556531560686,
        "avg_time": 4862.114134834187,
        "null_rate": 24.970999978410685,
    },
    "description": {
        "avg_changes": 0.00994849738993504,
        "avg_time": 4075.7180196719555,
        "null_rate": 96.05276996836581,
    },
    "dewey_decimal_class": {
        "avg_changes": 0.00021771336721706,
        "avg_time": 2680.6572119054763,
        "null_rate": 76.66223740383693,
    },
    "download_url": {
        "avg_changes": 3.2627643407150594e-06,
        "avg_time": 3799.9510858689177,
        "null_rate": 99.9990396423993,
    },
    "edition_name": {
        "avg_changes": 0.016353884962166287,
        "avg_time": 2501.5990003755805,
        "null_rate": 84.06724080575512,
    },
    "first_sentence": {
        "avg_changes": 0.00046155171468105,
        "avg_time": 2870.9363006232848,
        "null_rate": 97.66133570909832,
    },
    "full_title": {
        "avg_changes": 0.00011503806475726,
        "avg_time": 3574.5778551611606,
        "null_rate": 83.2900569554944,
    },
    "genres": {
        "avg_changes": 0.0001099850539468249,
        "avg_time": 4225.712424710462,
        "null_rate": 95.42310541778477,
    },
    "ia_box_id": {
        "avg_changes": 0.00875898444275713,
        "avg_time": 4334.2403883045945,
        "null_rate": 98.58047153627967,
    },
    "ia_loaded_id": {
        "avg_changes": 0.00433774291246409,
        "avg_time": 4639.926917446104,
        "null_rate": 99.12732440394582,
    },
    "identfiers_doi": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.99987910849771,
    },
    "isbn": {
        "avg_changes": 5.864324298359673e-05,
        "avg_time": 4391.287071841471,
        "null_rate": 99.99738594641266,
    },
    "isbn_10": {
        "avg_changes": 0.00159044158387219,
        "avg_time": 2189.290347647592,
        "null_rate": 47.20149176975297,
    },
    "isbn_13": {
        "avg_changes": 0.00259165129364558,
        "avg_time": 2182.517485131629,
        "null_rate": 53.563249936395316,
    },
    "isbn_invalid": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.82886258027392,
    },
    "isbn_odd_length": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.99604873428878,
    },
    "language": {
        "avg_changes": 0.00011190499240271001,
        "avg_time": 3964.5964311839402,
        "null_rate": 99.98897503410751,
    },
    "language_code": {
        "avg_changes": 1.7393739485678797e-06,
        "avg_time": 5980.678061930792,
        "null_rate": 99.99951598179824,
    },
    "languages": {
        "avg_changes": 0.01864047809061937,
        "avg_time": 3566.2592093258927,
        "null_rate": 13.743305463482406,
    },
    "last_modified": {
        "avg_changes": 1.8759264397387323,
        "avg_time": 1147.029185088261,
        "null_rate": 0.0,
    },
    "latest_revision": {
        "avg_changes": 1.86341181975586,
        "avg_time": 1151.4106038215334,
        "null_rate": 14.434892714967434,
    },
    "lc_classification": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.99960669633417,
    },
    "lc_classifications": {
        "avg_changes": 0.10835713698067721,
        "avg_time": 3727.0948526132292,
        "null_rate": 55.0741091424873,
    },
    "lccn": {
        "avg_changes": 0.032324870449509394,
        "avg_time": 4457.0064625789955,
        "null_rate": 63.345670859848234,
    },
    "lexile": {
        "avg_changes": 0.00018394533774025,
        "avg_time": 3804.5748494750223,
        "null_rate": 99.99283542924871,
    },
    "links": {
        "avg_changes": 2.1230009394868718e-05,
        "avg_time": 4344.448730752096,
        "null_rate": 99.1056840266935,
    },
    "local_id": {
        "avg_changes": 0.09899996749834918,
        "avg_time": 3713.7805083007966,
        "null_rate": 90.28531534157707,
    },
    "location": {
        "avg_changes": 2.7147261106778217e-05,
        "avg_time": 4139.313417638165,
        "null_rate": 98.68507551766183,
    },
    "notes": {
        "avg_changes": 0.007310228049437831,
        "avg_time": 4049.597340419559,
        "null_rate": 58.737326911388955,
    },
    "number_of_pages": {
        "avg_changes": 0.02031132170435026,
        "avg_time": 3020.488160175053,
        "null_rate": 28.303092285506636,
    },
    "numer_of_pages": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.99908970308542,
    },
    "ocaid": {
        "avg_changes": 0.07212246118403358,
        "avg_time": 4536.284327194121,
        "null_rate": 86.09963861283981,
    },
    "oclc": {"avg_changes": 0.0, "avg_time": None, "null_rate": 99.99729498716427},
    "oclc_number": {
        "avg_changes": 1.1314080469074252e-05,
        "avg_time": 5163.568016187406,
        "null_rate": 98.42049875271157,
    },
    "oclc_numbers": {
        "avg_changes": 0.16214507130625946,
        "avg_time": 4415.5663334009805,
        "null_rate": 72.7867346529845,
    },
    "openlibrary": {
        "avg_changes": 1.0618411054909168e-05,
        "avg_time": 1565.1306130988191,
        "null_rate": 99.96893859778633,
    },
    "openstax_id": {
        "avg_changes": 2.1740504315759205e-06,
        "avg_time": 1088.3779651159664,
        "null_rate": 99.99984887581991,
    },
    "original_isbn": {
        "avg_changes": 4.193618230202721e-05,
        "avg_time": 4042.3220067167895,
        "null_rate": 99.98982146949095,
    },
    "other_titles": {
        "avg_changes": 0.0004928479420019201,
        "avg_time": 3045.3938752903773,
        "null_rate": 92.3030644555658,
    },
    "pagination": {
        "avg_changes": 0.012969567159618178,
        "avg_time": 3226.7455410927964,
        "null_rate": 38.3277465112487,
    },
    "physical_dimensions": {
        "avg_changes": 0.00165574165833615,
        "avg_time": 2336.1286827506556,
        "null_rate": 88.82399400935725,
    },
    "physical_format": {
        "avg_changes": 0.0252757864080486,
        "avg_time": 2990.290686930184,
        "null_rate": 74.53942242481871,
    },
    "providers": {
        "avg_changes": 1.0788965164304389e-05,
        "avg_time": 613.073642285292,
        "null_rate": 99.99788212301863,
    },
    "publish_country": {
        "avg_changes": 0.0017686773479458802,
        "avg_time": 5077.199110781045,
        "null_rate": 43.224561148297816,
    },
    "publish_date": {
        "avg_changes": 0.01194275096684623,
        "avg_time": 1554.2227079095205,
        "null_rate": 2.375328294689005,
    },
    "publish_places": {
        "avg_changes": 0.0057898217004331495,
        "avg_time": 1994.341504161254,
        "null_rate": 42.99474479698695,
    },
    "publishers": {
        "avg_changes": 0.02237455638050363,
        "avg_time": 3457.3003991145065,
        "null_rate": 3.8426022289713377,
    },
    "purchase_url": {
        "avg_changes": 4.639866354585907e-06,
        "avg_time": 3981.4979396974913,
        "null_rate": 99.99811781007027,
    },
    "revision": {
        "avg_changes": 1.875948197655085,
        "avg_time": 1147.0281870511515,
        "null_rate": 0.0,
    },
    "scan_on_demand": {
        "avg_changes": 3.4776351902837994e-07,
        "avg_time": 2898.1287552332524,
        "null_rate": 99.92408579528588,
    },
    "scan_records": {
        "avg_changes": 0.00012964799890521,
        "avg_time": 3205.4473507083862,
        "null_rate": 99.98468383572646,
    },
    "series": {
        "avg_changes": 0.00401957515766585,
        "avg_time": 3204.4294852015537,
        "null_rate": 80.22368259607113,
    },
    "source_records": {
        "avg_changes": 0.7901493616121843,
        "avg_time": 2860.2650136421976,
        "null_rate": 40.68450404134878,
    },
    "subject_people": {
        "avg_changes": 3.4776351902837994e-07,
        "avg_time": 662.1152364651041,
        "null_rate": 98.62561397958798,
    },
    "subject_place": {
        "avg_changes": 2.471291105868943e-05,
        "avg_time": 5342.185810704979,
        "null_rate": 89.61097153597943,
    },
    "subject_places": {
        "avg_changes": 8.704289529433181e-07,
        "avg_time": 381.2970664955325,
        "null_rate": 96.05146727593328,
    },
    "subject_time": {
        "avg_changes": 2.263168055640368e-06,
        "avg_time": 4973.765566830855,
        "null_rate": 97.21149746241882,
    },
    "subject_times": {
        "avg_changes": 3.482787349256266e-07,
        "avg_time": 438.968282900625,
        "null_rate": 98.83034203735349,
    },
    "subjects": {
        "avg_changes": 0.03449360606439241,
        "avg_time": 4599.300899736925,
        "null_rate": 37.794880971557845,
    },
    "subtitle": {
        "avg_changes": 0.004560643398696529,
        "avg_time": 2482.717633309411,
        "null_rate": 59.20158807514798,
    },
    "table_of_contents": {
        "avg_changes": 0.032701508030608215,
        "avg_time": 2178.1050423741062,
        "null_rate": 96.16404534155853,
    },
    "title": {
        "avg_changes": 0.0662905959070669,
        "avg_time": 4760.336095255412,
        "null_rate": 0.007921608094920686,
    },
    "title_prefix": {
        "avg_changes": 0.04906178010431137,
        "avg_time": 5086.026896559968,
        "null_rate": 93.72749750541196,
    },
    "translated_from": {
        "avg_changes": 0.0010410571170480598,
        "avg_time": 2004.8195190569927,
        "null_rate": 99.85105315709933,
    },
    "translation_of": {
        "avg_changes": 0.00088546562758017,
        "avg_time": 2024.7585332277517,
        "null_rate": 99.92523656153789,
    },
    "type_key": {"avg_changes": 0.0, "avg_time": None, "null_rate": 0.0},
    "uri_descriptions": {
        "avg_changes": 0.00369119337377338,
        "avg_time": 5404.605885777111,
        "null_rate": 98.712848373139,
    },
    "uris": {
        "avg_changes": 0.00372321614565784,
        "avg_time": 5390.546379631029,
        "null_rate": 98.70567169031901,
    },
    "url": {
        "avg_changes": 8.874087132452107e-06,
        "avg_time": 4857.297609648498,
        "null_rate": 97.73399530375015,
    },
    "volumes": {
        "avg_changes": 5.224789519709643e-06,
        "avg_time": 582.3981145121373,
        "null_rate": 99.99868563770113,
    },
    "weight": {
        "avg_changes": 0.00126623703007436,
        "avg_time": 1264.6373365631193,
        "null_rate": 81.71725514578547,
    },
    "word_count": {
        "avg_changes": 0.0,
        "avg_time": None,
        "null_rate": 99.99871096001995,
    },
    "work_title": {
        "avg_changes": 2.784695053743439e-05,
        "avg_time": 5483.656414422365,
        "null_rate": 98.91260230617489,
    },
    "work_titles": {
        "avg_changes": 4.9250866263229766e-05,
        "avg_time": 4044.6118644717003,
        "null_rate": 98.18311092026931,
    },
    "works": {
        "avg_changes": 0.423037923586201,
        "avg_time": 3865.3224558852153,
        "null_rate": 20.85877016971876,
    },
}
