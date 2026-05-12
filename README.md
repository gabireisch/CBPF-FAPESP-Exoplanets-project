Semana 1 - pegando TIC IDs do TESS;

dados_tess.py le o csv dadosTESS baixado de https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI e separa em TIC IDs  positivos e negativos (csv). (baixei também o ExoFOP positivos mas são apenas 52);

teste.py pega os TIC IDs do csv e salva a curva de luz em .FITS;


Semana 2 - curvas de luz em .FITS;

LCs_positivos.py e LCs_negativos.py leem os respectivos csvs e baixam a curva de luz para cada TIC ID do csv.
