% Parte utilizzata per accesso ai dati grezzi
get_temp(X, T)      :- meteo_data(X, T, _, _, _, _, _, _, _, _).
get_rh(X, H)        :- meteo_data(X, _, H, _, _, _, _, _, _, _).
get_wind100(X, V)   :- meteo_data(X, _, _, _, _, V, _, _, _, _).
get_gusts(X, G)     :- meteo_data(X, _, _, _, _, _, _, _, G, _).
get_wind_dir100(X, D) :- meteo_data(X, _, _, _, _, _, _, D, _, _).
get_wind_dir10(X, D)  :- meteo_data(X, _, _, _, _, _, D, _, _, _).

% Trasforma la velocità del vento in una percentuale di potenza producibile
calc_wind_potential(V, P) :-
    V < 3.0, P is 0.0, !.
calc_wind_potential(V, P) :-
    V > 25.0, P is 0.0, !.
calc_wind_potential(V, P) :-
    V >= 12.0, P is 1.0, !.
calc_wind_potential(V, P) :-
    Base is (V - 3.0) / 9.0,
    P is Base * Base * Base.

get_wind_score(X, Score) :-
    get_wind100(X, V),
    calc_wind_potential(V, Score).

% Le turbine producono più energia quando l aria è fredda
get_density_score(X, Raw) :-
    get_temp(X, T),
    Diff is 15.0 - T,
    Correction is Diff * 0.003,
    Raw is 1.0 + Correction.

% Indice di Turbolenza
get_turbulence_idx(X, Index) :-
    get_gusts(X, G), get_wind100(X, V),
    V > 1.0,
    Index is G / V, !.
get_turbulence_idx(_, 0.0).

% Misura quanto cambia la direzione del vento tra la base e la cima della turbina.
get_shear_index(X, DiffDegrees) :-
    get_wind_dir10(X, D10), get_wind_dir100(X, D100),
    Delta is abs(D100 - D10),
    DiffDegrees is min(Delta, 360.0 - Delta).

% Calcola il rischio che si formi ghiaccio sulle pale
get_icing_prob(X, RiskScore) :-
    get_temp(X, T), get_rh(X, H),
    T < 3.0,
    H > 80.0,
    % Più bassa è T, più alto è il rischio: (3 - T)
    % Più alta è H, più alto è il rischio: (H - 80)
    T_factor is (3.0 - T),
    H_factor is (H - 80.0),
    RiskScore is T_factor * H_factor, !.
get_icing_prob(_, 0.0).

% Restituisce i valori
kb_extract_features(X, WindScore, TurbIdx, DensScore, IcingRisk, ShearDeg) :-
    get_wind_score(X, WindScore),
    get_turbulence_idx(X, TurbIdx),
    get_density_score(X, DensScore),
    get_icing_prob(X, IcingRisk),
    get_shear_index(X, ShearDeg).