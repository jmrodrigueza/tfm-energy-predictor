import unicodedata


# Function to normalize the text. Given a text, it returns the text in lowercase and without accents
def normalize_text(text: str) -> str:
    text = text.lower()
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    return text


# The dictionary with the regions and their corresponding CCAA not normalized
region_ccaa = {
    'A Coruña': 'Galicia',
    'Araba/Alava': 'País Vasco',
    'Albacete': 'Castilla-La Mancha',
    'Alicante': 'Comunidad Valenciana',
    'Almería': 'Andalucía',
    'Asturias': 'Asturias',
    'Ávila': 'Castilla y León',
    'Badajoz': 'Extremadura',
    'Baleares': 'Islas Baleares',
    'Illes Balears': 'Islas Baleares',
    'Barcelona': 'Cataluña',
    'Burgos': 'Castilla y León',
    'Cáceres': 'Extremadura',
    'Cádiz': 'Andalucía',
    'Cantabria': 'Cantabria',
    'Castellón': 'Comunidad Valenciana',
    'Ciudad Real': 'Castilla-La Mancha',
    'Córdoba': 'Andalucía',
    'Cuenca': 'Castilla-La Mancha',
    'Girona': 'Cataluña',
    'Granada': 'Andalucía',
    'Guadalajara': 'Castilla-La Mancha',
    'Gipuzkoa': 'País Vasco',
    'Huelva': 'Andalucía',
    'Huesca': 'Aragón',
    'Jaén': 'Andalucía',
    'La Rioja': 'La Rioja',
    'Las Palmas': 'Canarias',
    'León': 'Castilla y León',
    'Lleida': 'Cataluña',
    'Lugo': 'Galicia',
    'Madrid': 'Comunidad de Madrid',
    'Málaga': 'Andalucía',
    'Murcia': 'Región de Murcia',
    'Navarra': 'Navarra',
    'Ourense': 'Galicia',
    'Palencia': 'Castilla y León',
    'Pontevedra': 'Galicia',
    'Salamanca': 'Castilla y León',
    'Sta. Cruz de Tenerife': 'Canarias',
    'Segovia': 'Castilla y León',
    'Sevilla': 'Andalucía',
    'Soria': 'Castilla y León',
    'Tarragona': 'Cataluña',
    'Teruel': 'Aragón',
    'Toledo': 'Castilla-La Mancha',
    'Valencia': 'Comunidad Valenciana',
    'Valladolid': 'Castilla y León',
    'Bizkaia': 'País Vasco',
    'Zamora': 'Castilla y León',
    'Zaragoza': 'Aragón',
    'Ceuta': 'Ceuta',
    'Melilla': 'Melilla'
}

# Id_statión and province mapping
id_station_province = {
    'C439J': 'STA. CRUZ DE TENERIFE',
    '5612B': 'SEVILLA',
    '3094B': 'CUENCA',
    '9394X': 'ZARAGOZA',
    'B434X': 'ILLES BALEARS',
    '8293X': 'VALENCIA',
    '2755X': 'ZAMORA',
    '2400E': 'PALENCIA',
    '8500A': 'CASTELLON',
    'C249I': 'LAS PALMAS',
    '2462': 'MADRID',
    '9001D': 'CANTABRIA',
    '5047E': 'GRANADA',
    '9563X': 'CASTELLON',
    '9573X': 'TERUEL',
    '4358X': 'BADAJOZ',
    '1387E': 'A CORUÑA',
    '1212E': 'ASTURIAS',
    '0016A': 'TARRAGONA',
    'C447A': 'STA. CRUZ DE TENERIFE',
    '9434': 'ZARAGOZA',
    '6293X': 'ALMERIA',
    '8050X': 'ALICANTE',
    '5972X': 'CADIZ',
    '2235U': 'PALENCIA',
    '3260B': 'TOLEDO',
    '9051': 'BURGOS',
    '0367': 'GIRONA',
    '3175': 'MADRID',
    '6001': 'CADIZ',
    '9981A': 'TARRAGONA',
    '1331A': 'ASTURIAS',
    '8489X': 'CASTELLON',
    '6084X': 'MALAGA',
    '5246': 'JAEN',
    '1183X': 'ASTURIAS',
    '7178I': 'MURCIA',
    '9771C': 'LLEIDA',
    '4067': 'TOLEDO',
    '3365A': 'TOLEDO',
    '9784P': 'HUESCA',
    '0149X': 'BARCELONA',
    '5641X': 'SEVILLA',
    '6156X': 'MALAGA',
    'C459Z': 'STA. CRUZ DE TENERIFE',
    '9019B': 'CANTABRIA',
    '5995B': 'CADIZ',
    '7096B': 'ALBACETE',
    '0252D': 'BARCELONA',
    '9262': 'NAVARRA',
    '1083L': 'CANTABRIA',
    '1208H': 'ASTURIAS',
    '5427X': 'CORDOBA',
    '6277B': 'ALMERIA',
    '1631E': 'OURENSE',
    '1700X': 'OURENSE',
    '9569A': 'TERUEL',
    'C029O': 'LAS PALMAS',
    '2298': 'BURGOS',
    '5514': 'GRANADA',
    '9390': 'ZARAGOZA',
    '4220X': 'CIUDAD REAL',
    '7119B': 'MURCIA',
    '5192': 'JAEN',
    '1221D': 'ASTURIAS',
    '9201K': 'HUESCA',
    '1111': 'CANTABRIA',
    '6325O': 'ALMERIA',
    '1014': 'GIPUZKOA',
    '1002Y': 'NAVARRA',
    '3111D': 'MADRID',
    '6058I': 'MALAGA',
    '0324A': 'GIRONA',
    '8325X': 'VALENCIA',
    '4148': 'CIUDAD REAL',
    '1159': 'CANTABRIA',
    '6000A': 'MELILLA',
    '4452': 'BADAJOZ',
    'C659M': 'LAS PALMAS',
    'C629X': 'LAS PALMAS',
    '1283U': 'ASTURIAS',
    '4244X': 'BADAJOZ',
    '4410X': 'BADAJOZ',
    '9263D': 'NAVARRA',
    '9898': 'HUESCA',
    '1111X': 'CANTABRIA',
    '9208E': 'HUESCA',
    '3576X': 'CACERES',
    '2946X': 'SALAMANCA',
    '7012C': 'MURCIA',
    '5270B': 'JAEN',
    '4642E': 'HUELVA',
    'C139E': 'STA. CRUZ DE TENERIFE',
    '9170': 'LA RIOJA',
    '1082': 'BIZKAIA',
    '3168D': 'GUADALAJARA',
    '4560Y': 'HUELVA',
    '4103X': 'CIUDAD REAL',
    '7275C': 'MURCIA',
    '9698U': 'LLEIDA',
    '1351': 'A CORUÑA',
    '8414A': 'VALENCIA',
    '3200': 'MADRID',
    '7031': 'MURCIA',
    '9434P': 'ZARAGOZA',
    '1014A': 'GIPUZKOA',
    'B691Y': 'ILLES BALEARS',
    '4549Y': 'HUELVA',
    '4267X': 'CORDOBA',
    '2737E': 'LEON',
    '9111': 'BURGOS',
    'C429I': 'STA. CRUZ DE TENERIFE',
    '3391': 'AVILA',
    '8501': 'CASTELLON',
    '2539': 'VALLADOLID',
    '9091R': 'ARABA/ALAVA',
    '1059X': 'BIZKAIA',
    '5181D': 'JAEN',
    '6302A': 'ALMERIA',
    '5860E': 'HUELVA',
    'B013X': 'ILLES BALEARS',
    '1484C': 'PONTEVEDRA',
    '8175': 'ALBACETE',
    '2661': 'LEON',
    '3195': 'MADRID',
    '9263X': 'NAVARRA',
    'C229J': 'LAS PALMAS',
    '8416Y': 'VALENCIA',
    '5390Y': 'CORDOBA',
    '2916A': 'SALAMANCA',
    '2775X': 'ZAMORA',
    '2374X': 'PALENCIA',
    '3013': 'GUADALAJARA',
    '1387': 'A CORUÑA',
    '0372C': 'GIRONA',
    '1735X': 'OURENSE',
    '3434X': 'CACERES',
    '1473A': 'A CORUÑA',
    '9244X': 'ZARAGOZA',
    'B228': 'ILLES BALEARS',
    '9091O': 'ARABA/ALAVA',
    '5973': 'CADIZ',
    '3196': 'MADRID',
    '6155A': 'MALAGA',
    '7247X': 'ALICANTE',
    '1041A': 'GIPUZKOA',
    '1475X': 'A CORUÑA',
    '3130C': 'GUADALAJARA',
    '3191E': 'MADRID',
    'B278': 'ILLES BALEARS',
    '5402': 'CORDOBA',
    '9619': 'LLEIDA',
    '1495': 'PONTEVEDRA',
    '5796': 'SEVILLA',
    '7228': 'MURCIA',
    '4511C': 'BADAJOZ',
    'C659H': 'LAS PALMAS',
    '6106X': 'MALAGA',
    '6205X': 'MALAGA',
    '1078I': 'BIZKAIA',
    '3526X': 'CACERES',
    '5910': 'CADIZ',
    'B893': 'ILLES BALEARS',
    '0076': 'BARCELONA',
    '1024E': 'GIPUZKOA',
    '7002Y': 'MURCIA',
    '1037Y': 'GIPUZKOA',
    '9990X': 'LLEIDA',
    'C129Z': 'STA. CRUZ DE TENERIFE',
    '3338': 'MADRID',
    'C689E': 'LAS PALMAS',
    '2150H': 'SEGOVIA',
    '8368U': 'TERUEL',
    '7209': 'MURCIA',
    '1109': 'CANTABRIA',
    '4147X': 'CIUDAD REAL',
    '1210X': 'ASTURIAS',
    '2465': 'SEGOVIA',
    '3100B': 'MADRID',
    '2870': 'SALAMANCA',
    'C929I': 'STA. CRUZ DE TENERIFE',
    'C649I': 'LAS PALMAS',
    '2331': 'BURGOS',
    '0200E': 'BARCELONA',
    '4090Y': 'CUENCA',
    '1249X': 'ASTURIAS',
    '5298X': 'JAEN',
    '3266A': 'MADRID',
    '5911A': 'CADIZ',
    '3110C': 'MADRID',
    '5704B': 'SEVILLA',
    '9381I': 'TERUEL',
    '8178D': 'ALBACETE',
    '1428': 'A CORUÑA',
    '8177A': 'ALBACETE',
    '4386B': 'BADAJOZ',
    '2117D': 'BURGOS',
    '3519X': 'CACERES',
    '8309X': 'VALENCIA',
    'B248': 'ILLES BALEARS',
    '1400': 'A CORUÑA',
    '5960': 'CADIZ',
    '2614': 'ZAMORA',
    '3298X': 'TOLEDO',
    '2491C': 'SALAMANCA',
    '1057B': 'BIZKAIA',
    'B569X': 'ILLES BALEARS',
    '6367B': 'ALMERIA',
    '2422': 'VALLADOLID',
    '5530E': 'GRANADA',
    'B954': 'ILLES BALEARS',
    '2867': 'SALAMANCA',
    '2030': 'SORIA',
    '8025': 'ALICANTE',
    '7031X': 'MURCIA',
    '1207U': 'ASTURIAS',
    '1050J': 'GIPUZKOA',
    '1393': 'A CORUÑA',
    '7145D': 'MURCIA',
    '9294E': 'NAVARRA',
    '2444': 'AVILA',
    '4121': 'CIUDAD REAL',
    '9585': 'GIRONA',
    '3129': 'MADRID',
    '8416': 'VALENCIA',
    '5000C': 'CEUTA',
    '5582A': 'GRANADA',
    'C329Z': 'STA. CRUZ DE TENERIFE',
    '1505': 'LUGO',
    '1249I': 'ASTURIAS',
    '8019': 'ALICANTE',
    '1549': 'LEON',
    '8096': 'CUENCA',
    '5783': 'SEVILLA',
    'C430E': 'STA. CRUZ DE TENERIFE',
    '4061X': 'TOLEDO',
    '1542': 'ASTURIAS',
    '3469A': 'CACERES',
    'B341X': 'BALEARES',
    'B860X': 'ILLES BALEARS',
    'B644B': 'ILLES BALEARS',
    'B496X': 'ILLES BALEARS',
    'B301': 'ILLES BALEARS',
    'B103B': 'ILLES BALEARS',
    'B780X': 'ILLES BALEARS',
    'B760X': 'ILLES BALEARS',
    'B236C': 'ILLES BALEARS',
    'B362X': 'ILLES BALEARS',
    'B087X': 'ILLES BALEARS',
    'B614E': 'ILLES BALEARS',
    'B334X': 'ILLES BALEARS',
    'B605X': 'ILLES BALEARS',
    'B275E': 'BALEARES',
    'B410B': 'ILLES BALEARS',
    'B158X': 'ILLES BALEARS',
    'B640X': 'ILLES BALEARS',
    'B691': 'BALEARES',
    'B373X': 'ILLES BALEARS',
    'B526X': 'ILLES BALEARS',
    'B684A': 'ILLES BALEARS',
    'B603X': 'ILLES BALEARS',
    'B051A': 'ILLES BALEARS',
    'B800X': 'ILLES BALEARS',
    'B656A': 'ILLES BALEARS',
    'B662X': 'ILLES BALEARS',
    'B825B': 'ILLES BALEARS',
    'C019V': 'LAS PALMAS',
    'B870C': 'ILLES BALEARS',
    'C329B': 'STA. CRUZ DE TENERIFE',
    'C018J': 'LAS PALMAS',
    'C314Z': 'STA. CRUZ DE TENERIFE',
    'C117A': 'STA. CRUZ DE TENERIFE',
    'C316I': 'STA. CRUZ DE TENERIFE',
    'C248E': 'LAS PALMAS',
    'C419L': 'STA. CRUZ DE TENERIFE',
    'B986': 'ILLES BALEARS',
    'B957': 'ILLES BALEARS',
    'B908X': 'ILLES BALEARS',
    'C328W': 'STA. CRUZ DE TENERIFE',
    'C117Z': 'STA. CRUZ DE TENERIFE',
    'C317B': 'STA. CRUZ DE TENERIFE',
    'B925': 'ILLES BALEARS',
    'C038N': 'LAS PALMAS',
    'C428T': 'STA. CRUZ DE TENERIFE',
    'C239N': 'LAS PALMAS',
    'C258K': 'LAS PALMAS',
    'C129V': 'STA. CRUZ DE TENERIFE',
    'C101A': 'STA. CRUZ DE TENERIFE',
    'C319W': 'STA. CRUZ DE TENERIFE',
    'C048W': 'LAS PALMAS',
    'C406G': 'STA. CRUZ DE TENERIFE',
    'C419X': 'STA. CRUZ DE TENERIFE',
    'C126A': 'STA. CRUZ DE TENERIFE',
    'C148F': 'STA. CRUZ DE TENERIFE',
    'C468X': 'STA. CRUZ DE TENERIFE',
    'C839X': 'LAS PALMAS',
    'C629Q': 'LAS PALMAS',
    '0106X': 'BARCELONA',
    '0034X': 'TARRAGONA',
    'C649R': 'LAS PALMAS',
    'C656V': 'LAS PALMAS',
    'C917E': 'STA. CRUZ DE TENERIFE',
    'C458A': 'STA. CRUZ DE TENERIFE',
    'C614H': 'LAS PALMAS',
    'C658L': 'LAS PALMAS',
    'C446G': 'STA. CRUZ DE TENERIFE',
    'C648N': 'LAS PALMAS',
    'C939T': 'STA. CRUZ DE TENERIFE',
    '0120X': 'BARCELONA',
    'C635B': 'LAS PALMAS',
    'C639U': 'LAS PALMAS',
    '0092X': 'BARCELONA',
    'C658X': 'LAS PALMAS',
    '0016B': 'TARRAGONA',
    '0061X': 'BARCELONA',
    'C928I': 'STA. CRUZ DE TENERIFE',
    'C648C': 'LAS PALMAS',
    'C619Y': 'LAS PALMAS',
    'C611E': 'LAS PALMAS',
    '0009X': 'TARRAGONA',
    'C628B': 'LAS PALMAS',
    'C639M': 'LAS PALMAS',
    'C625O': 'LAS PALMAS',
    '0066X': 'BARCELONA',
    '0042Y': 'TARRAGONA',
    'C619X': 'LAS PALMAS',
    '0114X': 'BARCELONA',
    'C623I': 'LAS PALMAS',
    'C457I': 'STA. CRUZ DE TENERIFE',
    'C916Q': 'STA. CRUZ DE TENERIFE',
    'C669B': 'LAS PALMAS',
    'C925F': 'STA. CRUZ DE TENERIFE',
    'C612F': 'LAS PALMAS',
    'C668V': 'LAS PALMAS',
    '0073X': 'BARCELONA',
    'C438N': 'STA. CRUZ DE TENERIFE',
    'C449F': 'STA. CRUZ DE TENERIFE',
    'C665T': 'LAS PALMAS',
    '0421X': 'GIRONA',
    '1074C': 'BIZKAIA',
    '0222X': 'BARCELONA',
    '1064L': 'BIZKAIA',
    '0171X': 'BARCELONA',
    '1021X': 'GIPUZKOA',
    '1060X': 'ARABA/ALAVA',
    '0421E': 'GIRONA',
    '0244X': 'BARCELONA',
    '1135C': 'CANTABRIA',
    '0312X': 'GIRONA',
    '0281Y': 'GIRONA',
    '1044X': 'ARABA/ALAVA',
    '0341X': 'BARCELONA',
    '1010X': 'NAVARRA',
    '1048X': 'GIPUZKOA',
    '1037X': 'GIPUZKOA',
    '1012P': 'GIPUZKOA',
    '0429X': 'GIRONA',
    '1124E': 'CANTABRIA',
    '1069Y': 'BIZKAIA',
    '0149D': 'BARCELONA',
    '0433D': 'GIRONA',
    '0194D': 'BARCELONA',
    '1056K': 'BIZKAIA',
    '1025X': 'GIPUZKOA',
    '1038X': 'GIPUZKOA',
    '0370E': 'GIRONA',
    '0360X': 'GIRONA',
    '0260X': 'BARCELONA',
    '1096X': 'CANTABRIA',
    '0294B': 'GIRONA',
    '0158O': 'BARCELONA',
    '1167B': 'CANTABRIA',
    '1109X': 'CANTABRIA',
    '0158X': 'BARCELONA',
    '0413A': 'GIRONA',
    '0411X': 'GIRONA',
    '1154H': 'CANTABRIA',
    '0284X': 'GIRONA',
    '0363X': 'GIRONA',
    '0349': 'BARCELONA',
    '1089U': 'CANTABRIA',
    '0394X': 'GIRONA',
    '0385X': 'GIRONA',
    '0320I': 'GIRONA',
    '1021Y': 'NAVARRA',
    '1025A': 'GIPUZKOA',
    '1078C': 'BIZKAIA',
    '1033X': 'NAVARRA',
    '1052A': 'GIPUZKOA',
    '1026X': 'GIPUZKOA',
    '1049N': 'GIPUZKOA',
    '0341': 'BARCELONA',
    '1152C': 'CANTABRIA',
    '1083B': 'BIZKAIA',
    '1354C': 'A CORUÑA',
    '1477V': 'PONTEVEDRA',
    '1541B': 'LEON',
    '1390X': 'A CORUÑA',
    '1521X': 'LUGO',
    '1234P': 'ASTURIAS',
    '1279X': 'ASTURIAS',
    '1406X': 'A CORUÑA',
    '1561I': 'LEON',
    '1226X': 'ASTURIAS',
    '1476R': 'A CORUÑA',
    '1521I': 'LUGO',
    '1455I': 'PONTEVEDRA',
    '1486E': 'PONTEVEDRA',
    '1477U': 'PONTEVEDRA',
    '1174I': 'CANTABRIA',
    '1302F': 'ASTURIAS',
    '1639X': 'OURENSE',
    '1410X': 'A CORUÑA',
    '1468X': 'PONTEVEDRA',
    '1466A': 'PONTEVEDRA',
    '1496X': 'PONTEVEDRA',
    '1179B': 'ASTURIAS',
    '1583X': 'OURENSE',
    '1399': 'A CORUÑA',
    '1309C': 'ASTURIAS',
    '1347T': 'LUGO',
    '1435C': 'A CORUÑA',
    '1387D': 'A CORUÑA',
    '1489A': 'PONTEVEDRA',
    '1465U': 'PONTEVEDRA',
    '1176A': 'CANTABRIA',
    '1276F': 'ASTURIAS',
    '1341B': 'ASTURIAS',
    '1272B': 'ASTURIAS',
    '1186P': 'ASTURIAS',
    '1297E': 'LUGO',
    '1223P': 'ASTURIAS',
    '1327A': 'ASTURIAS',
    '1203D': 'ASTURIAS',
    '1518A': 'LUGO',
    '1178Y': 'LEON',
    '1446X': 'LUGO',
    '1442U': 'A CORUÑA',
    '1344X': 'LUGO',
    '1199X': 'ASTURIAS',
    '2182C': 'SEGOVIA',
    '2453E': 'AVILA',
    '1679A': 'LUGO',
    '2048A': 'SORIA',
    '2401X': 'PALENCIA',
    '2084Y': 'SORIA',
    '1701X': 'OURENSE',
    '1740': 'CANTABRIA',
    '2276B': 'PALENCIA',
    '2482B': 'SEGOVIA',
    '2106B': 'BURGOS',
    '1730E': 'PONTEVEDRA',
    '2017Y': 'SORIA',
    '1719': 'PONTEVEDRA',
    '2290Y': 'BURGOS',
    '1706A': 'OURENSE',
    '2044B': 'SORIA',
    '2135A': 'SEGOVIA',
    '1738U': 'OURENSE',
    '2192C': 'SEGOVIA',
    '2456B': 'AVILA',
    '2503B': 'VALLADOLID',
    '2471Y': 'SEGOVIA',
    '2059B': 'SORIA',
    '2296A': 'SORIA',
    '1696O': 'OURENSE',
    '2311Y': 'BURGOS',
    '2166Y': 'VALLADOLID',
    '1658': 'LUGO',
    '2285B': 'BURGOS',
    '2092': 'SORIA',
    '2172Y': 'VALLADOLID',
    '2302N': 'BURGOS',
    '2005Y': 'SORIA',
    '2430Y': 'AVILA',
    '2362C': 'PALENCIA',
    '2243A': 'PALENCIA',
    '1723X': 'PONTEVEDRA',
    '2140A': 'SEGOVIA',
    '2096B': 'SORIA',
    '2885K': 'ZAMORA',
    '2918Y': 'SALAMANCA',
    '2624C': 'LEON',
    '2945A': 'SALAMANCA',
    '2978X': 'OURENSE',
    '2664B': 'LEON',
    '2847X': 'SALAMANCA',
    '2517A': 'VALLADOLID',
    '2611D': 'ZAMORA',
    '2742R': 'LEON',
    '3021Y': 'GUADALAJARA',
    '2568D': 'PALENCIA',
    '2789H': 'ZAMORA',
    '2828Y': 'AVILA',
    '2863C': 'SALAMANCA',
    '2626Y': 'LEON',
    '2966D': 'ZAMORA',
    '2891A': 'SALAMANCA',
    '2507Y': 'VALLADOLID',
    '2926B': 'SALAMANCA',
    '2512Y': 'AVILA',
    '2930Y': 'SALAMANCA',
    '2604B': 'VALLADOLID',
    '3085Y': 'GUADALAJARA',
    '2536D': 'ZAMORA',
    '2873X': 'SALAMANCA',
    '2555B': 'ZAMORA',
    '2882D': 'ZAMORA',
    '2969U': 'OURENSE',
    '3099Y': 'TOLEDO',
    '2593D': 'VALLADOLID',
    '2766E': 'ZAMORA',
    '2701D': 'LEON',
    '2777K': 'ZAMORA',
    '2804F': 'ZAMORA',
    '3040Y': 'CUENCA',
    '2914C': 'SALAMANCA',
    '2728B': 'LEON',
    '2565': 'ZAMORA',
    '3343Y': 'MADRID',
    '3209Y': 'GUADALAJARA',
    '3536X': 'CACERES',
    '3194Y': 'MADRID',
    '3125Y': 'MADRID',
    '3386A': 'CACERES',
    '3565X': 'CACERES',
    '3362Y': 'TOLEDO',
    '3126Y': 'MADRID',
    '3245Y': 'TOLEDO',
    '3182Y': 'MADRID',
    '3305Y': 'TOLEDO',
    '3268C': 'MADRID',
    '3337U': 'AVILA',
    '3254Y': 'TOLEDO',
    '3229Y': 'MADRID',
    '3547X': 'CACERES',
    '3503': 'CACERES',
    '3475X': 'CACERES',
    '3103': 'GUADALAJARA',
    '3436D': 'CACERES',
    '3422D': 'AVILA',
    '3170Y': 'MADRID',
    '3562X': 'CACERES',
    '3516X': 'CACERES',
    '3463Y': 'CACERES',
    '3531X': 'CACERES',
    '3104Y': 'MADRID',
    '3504X': 'CACERES',
    '3319D': 'AVILA',
    '3448X': 'CACERES',
    '3330Y': 'MADRID',
    '3423I': 'CACERES',
    '3540X': 'CACERES',
    '3427Y': 'TOLEDO',
    '3494U': 'CACERES',
    '3140Y': 'GUADALAJARA',
    '3512X': 'CACERES',
    '3455X': 'CACERES',
    '3514B': 'CACERES',
    '4520X': 'BADAJOZ',
    '4347X': 'CACERES',
    '4064Y': 'CIUDAD REAL',
    '4091Y': 'ALBACETE',
    '4499X': 'BADAJOZ',
    '4095Y': 'CUENCA',
    '4411C': 'CACERES',
    '4489X': 'BADAJOZ',
    '4051Y': 'CUENCA',
    '4193Y': 'CIUDAD REAL',
    '4325Y': 'BADAJOZ',
    '4236Y': 'CACERES',
    '4486X': 'BADAJOZ',
    '4427X': 'BADAJOZ',
    '4478X': 'BADAJOZ',
    '4138Y': 'CIUDAD REAL',
    '4300Y': 'CIUDAD REAL',
    '4116I': 'CIUDAD REAL',
    '4245X': 'CACERES',
    '4075Y': 'CUENCA',
    '4260': 'BADAJOZ',
    '4492F': 'BADAJOZ',
    '4340': 'BADAJOZ',
    '4210Y': 'CIUDAD REAL',
    '4464X': 'BADAJOZ',
    '4007Y': 'ALBACETE',
    '4263X': 'CORDOBA',
    '4339X': 'CACERES',
    '4395X': 'BADAJOZ',
    '4527X': 'HUELVA',
    '4093Y': 'CUENCA',
    '4497X': 'BADAJOZ',
    '4070Y': 'CUENCA',
    '4468X': 'BADAJOZ',
    '4501X': 'BADAJOZ',
    '4089A': 'CUENCA',
    '4096Y': 'ALBACETE',
    '4436Y': 'BADAJOZ',
    '5612X': 'SEVILLA',
    '5702X': 'SEVILLA',
    '5165X': 'JAEN',
    '5429X': 'CORDOBA',
    '5515X': 'GRANADA',
    '5361X': 'CORDOBA',
    '5625X': 'CORDOBA',
    '5164B': 'JAEN',
    '4575X': 'HUELVA',
    '5346X': 'CORDOBA',
    '5515D': 'GRANADA',
    '4554X': 'HUELVA',
    '5473X': 'BADAJOZ',
    '5107D': 'GRANADA',
    '5304Y': 'CIUDAD REAL',
    '4541X': 'HUELVA',
    '5598X': 'CORDOBA',
    '5394X': 'CORDOBA',
    '5060X': 'ALMERIA',
    '5406X': 'JAEN',
    '5279X': 'JAEN',
    '5514Z': 'GRANADA',
    '5038Y': 'JAEN',
    '5281X': 'JAEN',
    '5470': 'CORDOBA',
    '5516D': 'GRANADA',
    '5656': 'SEVILLA',
    '5341C': 'CIUDAD REAL',
    '5412X': 'CORDOBA',
    '4622X': 'HUELVA',
    '5210X': 'JAEN',
    '5654X': 'SEVILLA',
    '4584X': 'HUELVA',
    '5624X': 'CORDOBA',
    '6143X': 'MALAGA',
    '6040X': 'MALAGA',
    '6088X': 'MALAGA',
    '6069X': 'MALAGA',
    '6281X': 'GRANADA',
    '6258X': 'GRANADA',
    '6201X': 'MALAGA',
    '6032X': 'MALAGA',
    '6172X': 'MALAGA',
    '6042I': 'CADIZ',
    '5726X': 'SEVILLA',
    '5891X': 'SEVILLA',
    '5788X': 'SEVILLA',
    '5790Y': 'SEVILLA',
    '6213X': 'MALAGA',
    '6050X': 'MALAGA',
    '6199X': 'MALAGA',
    '6268Y': 'GRANADA',
    '5858X': 'HUELVA',
    '5919X': 'CADIZ',
    '5906X': 'CADIZ',
    '6056X': 'CADIZ',
    '6045X': 'MALAGA',
    '6127X': 'MALAGA',
    '5769X': 'HUELVA',
    '6175X': 'MALAGA',
    '5733X': 'SEVILLA',
    '5950X': 'CADIZ',
    '6057X': 'MALAGA',
    '5835X': 'SEVILLA',
    '5996B': 'CADIZ',
    '6100B': 'MALAGA',
    '5983X': 'CADIZ',
    '6267X': 'GRANADA',
    '5941X': 'CADIZ',
    '6272X': 'GRANADA',
    '6083X': 'MALAGA',
    '6076X': 'MALAGA',
    '5998X': 'SEVILLA',
    '7012D': 'MURCIA',
    '7019X': 'MURCIA',
    '7007Y': 'MURCIA',
    '7067Y': 'ALBACETE',
    '6332Y': 'ALMERIA',
    '7072Y': 'ALBACETE',
    '6307X': 'ALMERIA',
    '7026X': 'MURCIA',
    '6375X': 'MALAGA',
    '7023X': 'MURCIA',
    '6364X': 'ALMERIA',
    '7020C': 'MURCIA',
    '7066Y': 'ALBACETE',
    '6291B': 'ALMERIA',
    '6329X': 'ALMERIA',
    '6340X': 'ALMERIA',
    '7250C': 'MURCIA',
    '7218Y': 'MURCIA',
    '7121A': 'MURCIA',
    '7227X': 'MURCIA',
    '7195X': 'MURCIA',
    '8008Y': 'ALICANTE',
    '7244X': 'ALICANTE',
    '7172X': 'MURCIA',
    '7203A': 'MURCIA',
    '7237E': 'MURCIA',
    '8013X': 'ALICANTE',
    '7158X': 'MURCIA',
    '7261X': 'ALICANTE',
    '8005X': 'VALENCIA',
    '7080X': 'MURCIA',
    '7211B': 'MURCIA',
    '7138B': 'MURCIA',
    '7103Y': 'ALBACETE',
    '7127X': 'MURCIA',
    '8018X': 'ALICANTE',
    '8270X': 'VALENCIA',
    '8376': 'TERUEL',
    '8300X': 'VALENCIA',
    '8354X': 'TERUEL',
    '8245Y': 'CUENCA',
    '8381X': 'VALENCIA',
    '8337X': 'VALENCIA',
    '8059C': 'ALICANTE',
    '8193E': 'VALENCIA',
    '8395X': 'VALENCIA',
    '8283X': 'VALENCIA',
    '8057C': 'ALICANTE',
    '8072Y': 'VALENCIA',
    '8058Y': 'VALENCIA',
    '8084Y': 'CUENCA',
    '8203O': 'VALENCIA',
    '8036Y': 'ALICANTE',
    '8155Y': 'CUENCA',
    '8492X': 'CASTELLON',
    '8446Y': 'VALENCIA',
    '9115X': 'LA RIOJA',
    '9001S': 'CANTABRIA',
    '8409X': 'VALENCIA',
    '8416X': 'VALENCIA',
    '9121X': 'LA RIOJA',
    '9012E': 'BURGOS',
    '8458X': 'TERUEL',
    '9027X': 'BURGOS',
    '9073X': 'ARABA/ALAVA',
    '8503Y': 'CASTELLON',
    '8486X': 'TERUEL',
    '9031C': 'BURGOS',
    '9016X': 'CANTABRIA',
    '8520X': 'CASTELLON',
    '9069C': 'BURGOS',
    '8472A': 'CASTELLON',
    '9060X': 'ARABA/ALAVA',
    '8439X': 'CASTELLON',
    '9218A': 'NAVARRA',
    '9145Y': 'LA RIOJA',
    '9207': 'HUESCA',
    '9228J': 'NAVARRA',
    '9171K': 'NAVARRA',
    '9228T': 'NAVARRA',
    '9195C': 'HUESCA',
    '9141V': 'LA RIOJA',
    '9257X': 'NAVARRA',
    '9252X': 'NAVARRA',
    '9188': 'LA RIOJA',
    '9198X': 'HUESCA',
    '9211F': 'HUESCA',
    '9136X': 'LA RIOJA',
    '9178X': 'ARABA/ALAVA',
    '9245X': 'NAVARRA',
    '9145X': 'ARABA/ALAVA',
    '9238X': 'NAVARRA',
    '9122I': 'ARABA/ALAVA',
    '9302Y': 'NAVARRA',
    '9280B': 'NAVARRA',
    '9427X': 'ZARAGOZA',
    '9374X': 'TERUEL',
    '9445L': 'HUESCA',
    '9299X': 'ZARAGOZA',
    '9436X': 'TERUEL',
    '9293X': 'LA RIOJA',
    '9336D': 'ZARAGOZA',
    '9262P': 'NAVARRA',
    '9287A': 'SORIA',
    '9321X': 'ZARAGOZA',
    '9354X': 'ZARAGOZA',
    '9352A': 'SORIA',
    '9451F': 'HUESCA',
    '9274X': 'NAVARRA',
    '9453X': 'HUESCA',
    '9301X': 'NAVARRA',
    '9344C': 'SORIA',
    '9377Y': 'GUADALAJARA',
    '9574B': 'ZARAGOZA',
    '9562X': 'CASTELLON',
    '9510X': 'ZARAGOZA',
    '9491X': 'HUESCA',
    '9657X': 'LLEIDA',
    '9501X': 'ZARAGOZA',
    '9590': 'LLEIDA',
    '9638D': 'LLEIDA',
    '9531Y': 'TERUEL',
    '9495Y': 'ZARAGOZA',
    '9647X': 'LLEIDA',
    '9550C': 'TERUEL',
    '9590D': 'LLEIDA',
    '9632X': 'LLEIDA',
    '9660': 'LLEIDA',
    '9546B': 'TERUEL',
    '9460X': 'HUESCA',
    '9561X': 'TERUEL',
    '9650X': 'LLEIDA',
    '9513X': 'TERUEL',
    '9808X': 'HUESCA',
    '9775X': 'LLEIDA',
    '9772X': 'LLEIDA',
    '9814X': 'HUESCA',
    '9689X': 'LLEIDA',
    '9843A': 'HUESCA',
    '9707': 'LLEIDA',
    '9726E': 'TARRAGONA',
    '9744B': 'LLEIDA',
    '9814I': 'HUESCA',
    '9838B': 'HUESCA',
    '9839V': 'HUESCA',
    '9776D': 'LLEIDA',
    '9724X': 'LLEIDA',
    '9677': 'LLEIDA',
    '9729X': 'LLEIDA',
    '9756X': 'HUESCA',
    '9751': 'HUESCA',
    '9718X': 'LLEIDA',
    '9947X': 'TARRAGONA',
    '9908X': 'HUESCA',
    '9911X': 'HUESCA',
    '9994X': 'LLEIDA',
    '9924X': 'HUESCA',
    '9961X': 'TARRAGONA',
    '9998X': 'TERUEL',
    '9894Y': 'HUESCA',
    '9901X': 'HUESCA',
    '9866C': 'HUESCA',
    '9855E': 'HUESCA',
    '9935X': 'TERUEL',
    '9988B': 'LLEIDA',
    '9975X': 'TARRAGONA',
    '9995Y': 'NAVARRA',
    '9918Y': 'HUESCA',
    'C453I': 'STA. CRUZ DE TENERIFE',
    'C468O': 'STA. CRUZ DE TENERIFE',
    'C449C': 'STA. CRUZ DE TENERIFE',
    'C418I': 'STA. CRUZ DE TENERIFE',
    'C426E': 'STA. CRUZ DE TENERIFE',
    'C426I': 'STA. CRUZ DE TENERIFE',
    'C448C': 'STA. CRUZ DE TENERIFE',
    'C436I': 'STA. CRUZ DE TENERIFE',
    'C415A': 'STA. CRUZ DE TENERIFE',
    'C422A': 'STA. CRUZ DE TENERIFE',
    'C458U': 'STA. CRUZ DE TENERIFE',
    'C456R': 'STA. CRUZ DE TENERIFE',
    'C418L': 'STA. CRUZ DE TENERIFE',
    'C426R': 'STA. CRUZ DE TENERIFE',
    'C919K': 'STA. CRUZ DE TENERIFE',
    'C467I': 'STA. CRUZ DE TENERIFE',
    'C455M': 'STA. CRUZ DE TENERIFE',
    'C449Q': 'STA. CRUZ DE TENERIFE',
    'C412N': 'STA. CRUZ DE TENERIFE',
    'C423R': 'STA. CRUZ DE TENERIFE',
    'C436L': 'STA. CRUZ DE TENERIFE',
    'C466O': 'STA. CRUZ DE TENERIFE',
    'C456E': 'STA. CRUZ DE TENERIFE',
    'C437E': 'STA. CRUZ DE TENERIFE',
    'C428U': 'STA. CRUZ DE TENERIFE',
    'C456P': 'STA. CRUZ DE TENERIFE',
    'C417J': 'STA. CRUZ DE TENERIFE',
    'C457E': 'STA. CRUZ DE TENERIFE',
    'C468I': 'STA. CRUZ DE TENERIFE',
    '0201X': 'BARCELONA',
    '1363X': 'A CORUÑA',
    '5459X': 'CORDOBA',
    '4608X': 'HUELVA',
    '4589X': 'HUELVA',
    '2734D': 'LEON',
    '2630X': 'LEON',
    '9946X': 'TARRAGONA',
    '1342X': 'LUGO',
    '8198Y': 'ALBACETE',
    '4362X': 'BADAJOZ',
    '0321': 'GIRONA',
    '8210Y': 'CUENCA',
    '1690A': 'OURENSE',
    '1103X': 'CANTABRIA',
    '3194U': 'MADRID',
    '0255B': 'BARCELONA',
    '9812M': 'HUESCA',
    '9201X': 'HUESCA',
    '0201D': 'BARCELONA',
    '4195E': 'CIUDAD REAL',
    '5103F': 'GRANADA',
    '6312E': 'ALMERIA',
    '6335O': 'ALMERIA',
    '6307C': 'ALMERIA',
    '1167G': 'CANTABRIA',
    '1178R': 'ASTURIAS',
    '1167J': 'CANTABRIA',
    'B398A': 'ILLES BALEARS',
    '5511': 'GRANADA',
    '5103E': 'GRANADA',
    '6248D': 'GRANADA',
    '6299I': 'GRANADA',
    '5051X': 'GRANADA',
}


# Normalize the region names and store them in a dictionary
normalized_regions = {normalize_text(k): v for k, v in region_ccaa.items()}


# Function to retrieve the CCAA of a region
def retrieve_ccaa(region: str) -> str:
    return normalized_regions.get(normalize_text(region), 'Provincia no encontrada')


def retrieve_province(id_station):
    return id_station_province.get(id_station, 'Unknown')
