# Notatki o danych

Ten plik zbiera to, co wiemy o danych meczu Barcelona – Athletic: co jest w plikach, jakie mają
wady i co z tego wynika dla rozwiązania. Dopisujcie tu kolejne obserwacje, najlepiej razem
z informacją, jak zostały zmierzone, żeby każdy mógł je sprawdzić.

Wszystkie liczby poniżej policzyliśmy na plikach z `data/processed/`, czyli po uruchomieniu
`src/convert_data.py`.

## Podstawowe pojęcia

**Klatka** to jeden obraz z wideo kamery taktycznej. Wideo ma 25 klatek na sekundę i 199 319
klatek, numerowanych od 0. Nasze zgłoszenie ma dokładnie jeden wiersz na każdą klatkę.

**Klatki oceniane** to te, na których organizatorzy liczą wynik, czyli dwie połowy meczu:

| część nagrania | klatki | czy oceniana |
|---|---|---|
| przed meczem | 0 – 11 184 | nie |
| pierwsza połowa | 11 185 – 90 651 | tak |
| przerwa | 90 652 – 110 807 | nie |
| druga połowa | 110 808 – 190 197 | tak |
| po meczu | 190 198 – 199 318 | nie |

Razem ocenianych jest 158 857 klatek. To, co wpiszemy poza nimi, nie ma żadnego znaczenia.

**Etykiety** są cztery: `H` oznacza, że piłkę ma Barcelona (gospodarz), `A` że ma ją Athletic
(gość), `D` że piłka jest martwa (gra stoi), a `X` że nie udzielamy odpowiedzi.

**Publiczne ground truth** to prawdziwe etykiety dla pierwszych 5 minut gry (klatki
11 185 – 18 684), które organizatorzy udostępnili w `evaluation/gt/`. Wolno na nie patrzeć, żeby
zrozumieć, co organizatorzy uznają za posiadanie i martwą piłkę. Nie wolno na nich trenować
modelu ani dobierać parametrów. W tych notatkach używamy ich tylko w pierwszym celu.

## Co jest w plikach

Organizatorzy dają cztery rodzaje plików, każdy w innym formacie. Skrypt `convert_data.py`
zamienia je na tabele Parquet, więc z surowymi plikami nie trzeba już pracować. Warto jednak
wiedzieć, co zawierają.

**Tracking (`*_atd.txt`)** to najważniejszy plik. Każda linia opisuje jedną klatkę: pozycję x, y
i prędkość każdego zawodnika oraz pozycję piłki. Tam, gdzie system nie widział zawodnika albo
piłki, wpisane jest `NaN`. Tracking powstał automatycznie z wideo i nikt go nie poprawiał, więc
zawiera błędy opisane niżej.

**Metadane trackingu (`*_atd.xml`)** mówią, który „tor” należy do której drużyny. Tor (w kodzie
`slot`) to jeden ciąg pozycji, który system śledzenia uznał za tego samego zawodnika. Torów
jest 46, a zawodników na boisku 22, bo system co jakiś czas gubi zawodnika i zaczyna dla niego
nowy tor. Tor nie jest więc konkretną osobą: wiemy tylko, do której drużyny należy. Tory 1–24
to Barcelona, a 25–46 to Athletic.

**Homografia (`*_homography.csv`)** to dla każdej klatki macierz, która przelicza pozycję na
boisku na piksel w obrazie i odwrotnie. Przyda się dopiero wtedy, gdy będziemy chcieli coś
narysować na wideo albo wykrywać piłkę z obrazu.

**Smart Tagging (`*_smart_tagging.xml`)** to 638 automatycznie wykrytych fragmentów gry, na
przykład „rozegranie od tyłu”, „piłka w polu karnym” albo „stały fragment” (rzut rożny, wolny,
od bramki, wrzut z autu). Każdy fragment ma początek, koniec i zwykle drużynę.

### Pułapki w surowych plikach

Konwersja do Parquet rozwiązuje pierwsze dwie z nich, ale trzeba o nich pamiętać, jeśli ktoś
sięgnie po surowe pliki.

- Surowy tracking numeruje klatki od 1, a zgłoszenie od 0. W tabelach Parquet numeracja jest
  już od 0.
- Surowe pozycje są ułamkiem boiska (od 0 do 1), a nie metrami. W tabelach Parquet są metry,
  przy boisku 105,3 × 68,0 m.
- Prędkość zawodników jest w metrach na sekundę, chociaż metadane w XML twierdzą, że
  w kilometrach na godzinę. Widać to po największej wartości w całym meczu, która wynosi 9,4.
  Tyle wynosi sprint w m/s, a w km/h byłby to trucht.
- Piłka ma w pliku trzecie pole (prędkość), ale jest ono prawie zawsze puste. W praktyce
  o piłce wiemy tylko, gdzie jest, a jej prędkość trzeba liczyć samemu z kolejnych pozycji.
- Pozycje czasem wychodzą trochę poza boisko, bo piłka i zawodnicy bywają za liniami. To nie
  błąd, tylko przydatna informacja (piłka za linią oznacza zwykle koniec gry).
- Metadane XML opisują cały plik jako jedną „połowę”, więc granice połów trzeba brać z tabeli
  powyżej, a nie z XML.
- W pierwszej połowie Barcelona atakuje od lewej do prawej.

## Piłka

Piłka to najważniejsza informacja w danych i jednocześnie najbardziej dziurawa.

**Piłki nie widać na 41% ocenianych klatek.** Jest widoczna na 59,3% z nich, tak samo w obu
połowach.

**Piłka nie „miga”, tylko znika na długo.** To najważniejsza obserwacja o piłce. Spodziewaliśmy
się wielu krótkich przerw, które da się uzupełnić, łącząc pozycję sprzed przerwy z pozycją po
niej. Jest inaczej: w całym ocenianym czasie jest tylko 150 przerw, przerw krótszych niż sekunda
prawie nie ma, a 85% wszystkich brakujących klatek leży w przerwach dłuższych niż 10 sekund.
19 przerw trwa ponad pół minuty. Z kolei kiedy piłka jest widoczna, to zwykle bez przerwy przez
dłuższy czas (typowo około 14 sekund).

Wniosek: uzupełnianie krótkich przerw nic nie da. Trzeba umieć zdecydować, co się dzieje przez
10–60 sekund, kiedy piłki w danych w ogóle nie ma.

**Piłka znika głównie pod bramkami.** Jeśli ostatnio była widziana w polu karnym, to brakuje jej
na 61–68% klatek. W pozostałych częściach boiska brakuje jej na około 27% klatek. Pod bramką jest
tłok i system częściej ją gubi.

**Zdarzają się błędne pozycje piłki.** Na około 200 klatkach piłka „przeskakuje” z prędkością
ponad 40 m/s, czyli szybciej niż najmocniejszy strzał. To pomyłki systemu (wziął za piłkę coś
innego) i trzeba je odfiltrować, zanim policzymy prędkość piłki.

**Piłka bywa za liniami boiska** na 4,2% klatek, na których ją widać.

## Zawodnicy

**Średnio widać 20,5 zawodnika na klatkę**, po około 10 z każdej drużyny, czyli zwykle kogoś
brakuje.

**Bywają duplikaty.** Na 8% klatek system widzi ponad 11 zawodników Barcelony, a na 5% ponad 11
zawodników Athleticu. Oznacza to, że tego samego zawodnika śledzi wtedy dwoma torami naraz.

**Bywa też, że brakuje wielu zawodników.** Na 9% klatek widać mniej niż 8 zawodników
którejś drużyny.

**Tory są krótkie.** Typowy tor trwa 14 sekund, a co dziesiąty krócej niż 2 sekundy. Nie da się
więc śledzić konkretnego zawodnika przez dłuższy czas i wszystko, co liczymy, musi opierać się
na drużynie, a nie na osobie.

**Zawodnik najbliżej piłki jest zwykle blisko, ale nie zawsze.** Na klatkach z widoczną piłką
najbliższy zawodnik jest typowo 1,9 m od niej. Na co czwartej klatce jest jednak dalej niż 4 m,
bo piłka leci po podaniu albo jest bezpańska.

## Martwa piłka (etykieta `D`)

Martwa piłka to czas, kiedy gra stoi: po aucie, faulu, golu, przed rzutem rożnym. W danych nie
ma pola, które by to mówiło, więc trzeba to rozpoznać po zachowaniu piłki i zawodników.

**Najlepszy sygnał to tempo zawodników.** Policzyliśmy średnią prędkość wszystkich widocznych
zawodników, wygładzoną w oknie 2 sekund. W publicznym ground truth wynosi ona 1,4 m/s, gdy
piłka jest martwa, i 2,2–2,6 m/s, gdy gra się toczy. Kiedy gra stoi, zawodnicy po prostu chodzą.
Ten sygnał działa także wtedy, gdy piłki nie widać.

**Większość czasu bez piłki w danych to przestój.** W całym meczu średnie tempo zawodników wynosi
2,0 m/s na klatkach z widoczną piłką i 1,0 m/s na klatkach bez niej. 80% klatek bez piłki ma
tempo poniżej 1,5 m/s, a wśród klatek z piłką takich jest tylko 26%. To dobra wiadomość: długie
przerwy w widoczności piłki w dużej części pokrywają się z martwą piłką, więc nie musimy
zgadywać, kto ma piłkę, tylko rozpoznać, że gra stoi.

**Martwa piłka wygląda w danych na kilka różnych sposobów.** W publicznych 5 minutach są cztery
takie odcinki i każdy jest inny:

| początek (klatka) | długość | co widać w danych |
|---|---|---|
| 12 851 | 5,5 s | piłka znika z danych, a zawodnicy zwalniają z 4 do 1,2 m/s |
| 15 889 | 20 s | piłka jest cały czas widoczna i na boisku: toczy się wzdłuż linii bocznej i wraca, a zawodnicy zwalniają do 0,8 m/s |
| 17 380 | 18 s | piłka znika na 15 s, po czym jest rzut wolny (jest na to tag) |
| 18 027 | 14,5 s | piłka wychodzi za linię końcową, po czym jest rzut od bramki (jest na to tag) |

Drugi wiersz jest najważniejszy. Pokazuje, że nie wystarczy sprawdzać, czy piłka wyszła poza
boisko albo zniknęła. Piłka bywa martwa, mimo że widać ją na boisku, na przykład po faulu.

**Martwa piłka kończy się w chwili wznowienia gry.** W danych widać to tak, że piłka nagle
rusza, a zawodnicy przyspieszają. Znalezienie tej chwili jest ważne, bo to jednocześnie początek
nowego posiadania.

## Posiadanie (etykiety `H` i `A`)

**Reguła „piłkę ma drużyna zawodnika, który jest najbliżej” myli się w co czwartej–piątej
klatce.** Na klatkach z widoczną piłką najbliższy zawodnik należy do drużyny, która według
ground truth ma piłkę, w około 80% przypadków dla Barcelony i 75% dla Athleticu. Reszta to
sytuacje, w których przeciwnik jest bliżej piłki, ale jej nie kontroluje: pressing, pojedynek,
piłka lecąca obok obrońcy. Sama odległość nie wystarczy.

**Posiadanie może być bardzo krótkie.** W publicznym ground truth są posiadania trwające 0,8 s,
1,3 s i 2,8 s. Nie możemy więc przyjąć, że drużyna musi utrzymać piłkę przez kilka sekund, żeby
liczyło się to jako przejęcie. Zbyt mocne wygładzanie wyników takie krótkie posiadania skasuje.

**Dla orientacji:** w publicznych 5 minutach jest 17 posiadań, średnio po 14,2 sekundy.

## Smart Tagging

Tagi brzmią jak gotowa odpowiedź (mają drużynę i czas), ale są dużo mniej dokładne, niż się
wydaje.

**Długość tagu jest sztuczna.** Wszystkie stałe fragmenty trwają dokładnie 35 sekund (kilka ma
5 sekund), a każdy tag „tworzenie sytuacji” trwa dokładnie 30 sekund. To znaczy, że system nie
zaznacza, kiedy coś się naprawdę zaczęło i skończyło, tylko wstawia okno o stałej długości
w okolicy zdarzenia. Samo wznowienie gry wypada w różnych miejscach tego okna: przy rzucie
wolnym z publicznych 5 minut około 12 sekund po początku tagu, a przy rzucie od bramki około
3 sekund.

**Wielu stałych fragmentów brakuje.** W całym meczu oznaczono ich tylko 59, w tym 5 rzutów
wolnych i 26 wrzutów z autu. W prawdziwym meczu jest ich wyraźnie więcej, więc nie można
zakładać, że brak tagu oznacza brak przerwy w grze.

**Wrzuty z autu i rozpoczęcia gry nie mają drużyny.** Pozostałe stałe fragmenty (rożne, wolne,
od bramki) mają.

**Tagi w pliku nie są ułożone po kolei w czasie**, więc przed użyciem trzeba je posortować po
kolumnie `start_frame`.

Wniosek: tag traktujemy jako podpowiedź „gdzieś tutaj było wznowienie gry i wykonywała je ta
drużyna”. Dokładną chwilę wznowienia trzeba znaleźć w trackingu.

## Plik z publicznym ground truth

Plik `evaluation/gt/BAR-ATH_possession_gt_public.csv` ma wiersz dla każdej klatki, tak jak
zgłoszenie, ale prawdziwe etykiety są tylko w klatkach 11 185 – 18 684 (od linii 11 187 pliku).
Wszystkie pozostałe wiersze mają `X`, co znaczy „tej klatki nie oceniamy”. Dlatego po otwarciu
pliku widać na początku same `X`.
