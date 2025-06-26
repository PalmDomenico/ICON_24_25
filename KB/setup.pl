:- include('knowledge_base.pl').
% Il vento � debole (< 4 m/s), spesso insufficiente per far girare le turbine in modo efficiente.
weak_wind(X) :-
    windspeed_100m(X, V),
    V < 4.

% Il vento � ottimale (8�14 m/s), ottimale per il funzionamento delle turbine con lo scopo di massimizzare la produzione.
optimal_wind(X) :-
    windspeed_100m(X, V),
    V >= 8,
    V =< 14.

% Umidit� critica (> 90%), pu� indicare condizioni atmosferiche instabili (nebbia, condensa, pioggia) esse influenzano le prestazioni.
critical_humidity(X) :-
    relativehumidity_2m(X, H),
    H > 90.

% Raffiche forti (> 10 m/s).
strong_gusts(X) :-
    windgusts_10m(X, G),
    G > 10.


% Vento ottimale, assenza di raffiche e umidit� elevata sono condizioni stabili che indicano una produzione continua e prevedibile.
stable_conditions(X) :-
    optimal_wind(X),
    \+ critical_humidity(X),
    \+ strong_gusts(X).


% Possibile bassa produzione, dovuta ad un vento troppo debole o condizioni atmosferiche sfavorevoli
low_expected_production(X) :-
    weak_wind(X);
    (strong_gusts(X), critical_humidity(X)).


meteo_info(RowId,
    Temp,
    RH,
    Dew,
    Wind10,
    Wind100,
    Dir10,
    Dir100,
    Gust,
    Power,
    Cluster) :-

    temperature_2m(RowId, Temp),
    relativehumidity_2m(RowId, RH),
    dewpoint_2m(RowId, Dew),
    windspeed_10m(RowId, Wind10),
    windspeed_100m(RowId, Wind100),
    winddirection_10m(RowId, Dir10),
    winddirection_100m(RowId, Dir100),
    windgusts_10m(RowId, Gust),
    power(RowId, Power),
    cluster(RowId, Cluster).