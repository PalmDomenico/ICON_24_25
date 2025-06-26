import pandas as pd
from pyswip import Prolog


def df_to_prolog_facts(facts_data, output_path):
    items = [
        "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
        "windspeed_10m", "windspeed_100m", "winddirection_10m",
        "winddirection_100m", "windgusts_10m", "Power"
    ]
    facts = {key: [] for key in items}

    for idx, row in facts_data.iterrows():
        row_id = f"r{idx}"
        facts["temperature_2m"].append(f"temperature_2m({row_id}, {row['temperature_2m']}).")
        facts["relativehumidity_2m"].append(f"relativehumidity_2m({row_id}, {row['relativehumidity_2m']}).")
        facts["dewpoint_2m"].append(f"dewpoint_2m({row_id}, {row['dewpoint_2m']}).")
        facts["windspeed_10m"].append(f"windspeed_10m({row_id}, {row['windspeed_10m']}).")
        facts["windspeed_100m"].append(f"windspeed_100m({row_id}, {row['windspeed_100m']}).")
        facts["winddirection_10m"].append(f"winddirection_10m({row_id}, {row['winddirection_10m']}).")
        facts["winddirection_100m"].append(f"winddirection_100m({row_id}, {row['winddirection_100m']}).")
        facts["windgusts_10m"].append(f"windgusts_10m({row_id}, {row['windgusts_10m']}).")
        facts["Power"].append(f"power({row_id}, {row['Power']}).")

    with open(output_path, 'w') as f:
        for key in facts:
            f.write("\n".join(facts[key]) + "\n\n")


def append_facts(output_path, labels):
    facts = []
    for i in range(len(labels)):
        row_id = f"r{i}"
        facts.append(f"cluster({row_id}, {labels[i]}).")
    with open(output_path, 'a') as f:
        for key in facts:
            f.write(f"{key}\n")


def add_rules(kb_file_path):
    rules = """
    % --- Rules ---
        % Il vento è debole (< 4 m/s), spesso insufficiente per far girare le turbine in modo efficiente.
        weak_wind(X) :-
            windspeed_100m(X, V),
            V < 4.
        
        % Il vento è ottimale (8–14 m/s), ottimale per il funzionamento delle turbine con lo scopo di massimizzare la produzione.
        optimal_wind(X) :-
            windspeed_100m(X, V),
            V >= 8,
            V =< 14.
        
        % Umidità critica (> 90%), può indicare condizioni atmosferiche instabili (nebbia, condensa, pioggia) esse influenzano le prestazioni.
        critical_humidity(X) :-
            relativehumidity_2m(X, H),
            H > 90.
        
        % Raffiche forti (> 10 m/s).
        strong_gusts(X) :-
            windgusts_10m(X, G),
            G > 10.
        
        
        % Vento ottimale, assenza di raffiche e umidità elevata sono condizioni stabili che indicano una produzione continua e prevedibile.
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



    """

    with open(kb_file_path, 'a') as f:
        f.write("\n\n" + rules.strip())


def query_prolog(prolog_file, predicate):
    prolog = Prolog()
    prolog.consult(prolog_file)
    results = list(prolog.query(f"{predicate}(X)"))
    return set(r['X'] for r in results)


def integrate_logical_features(csv_path, prolog_file):
    df = pd.read_csv(csv_path)
    num_rows = df.shape[0]

    features = {
        'weak_wind': query_prolog(prolog_file, 'weak_wind'),
        'optimal_wind': query_prolog(prolog_file, 'optimal_wind'),
        'critical_humidity': query_prolog(prolog_file, 'critical_humidity'),
        'strong_gusts': query_prolog(prolog_file, 'strong_gusts'),
        'stable_conditions': query_prolog(prolog_file, 'stable_conditions'),
        'low_expected_production': query_prolog(prolog_file, 'low_expected_production'),
    }

    for feature_name, matching_ids in features.items():
        df[feature_name] = [1 if f"r{i}" in matching_ids else 0 for i in range(num_rows)]

    df.to_csv(csv_path, index=False)
