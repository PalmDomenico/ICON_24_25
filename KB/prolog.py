from pyswip import Prolog
import os


def write_facts(df_raw, df_cluster, filename):
    if os.path.exists(filename):
        os.remove(filename)

    facts = []
    for row in df_raw.itertuples(index=True):
        try:
            idx = row.Index
            t = getattr(row, 'temperature_2m', 0)
            rh = getattr(row, 'relativehumidity_2m', 0)
            dew = getattr(row, 'dewpoint_2m', 0)
            w10 = getattr(row, 'windspeed_10m', 0)
            w100 = getattr(row, 'windspeed_100m', 0)
            d10 = getattr(row, 'winddirection_10m', 0)
            d100 = getattr(row, 'winddirection_100m', 0)
            g10 = getattr(row, 'windgusts_10m', 0)
            p = getattr(row, 'Power', 0)

            # meteo_data(id, temp, rh, dew, w10, w100, d10, d100, gust, power).
            facts.append(
                f"meteo_data(r{idx}, {t:.2f}, {rh:.2f}, {dew:.2f}, {w10:.2f}, {w100:.2f}, {d10:.2f}, {d100:.2f}, {g10:.2f}, {p:.2f}).")
        except:
            continue

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(facts))
        f.write('\n')

        # Write data cluster
        if 'Cluster' in df_cluster.columns:
            cluster_facts = []
            for row in df_cluster.itertuples(index=True):
                cluster_facts.append(f"cluster(r{row.Index}, {int(row.Cluster)}).")
            f.write('\n'.join(cluster_facts))


def query_kb_features(df_target, rules_file, facts_file):
    prolog = Prolog()

    try:
        prolog.consult(rules_file)
        prolog.consult(facts_file)
    except:
        return df_target

    # Query unica
    query_str = "kb_extract_features(X, Wind, Turb, Dens, Ice, Shear)"
    results = list(prolog.query(query_str))

    maps = {
        'KB_WindPotential': {},
        'KB_Turbulence': {},
        'KB_DensityScore': {},
        'KB_IcingProb': {},
        'KB_Shear': {}
    }

    for res in results:
        row_key = str(res['X'])
        if row_key.startswith('r'):
            try:
                idx = int(row_key[1:])
                maps['KB_WindPotential'][idx] = float(res['Wind'])
                maps['KB_Turbulence'][idx] = float(res['Turb'])
                maps['KB_DensityScore'][idx] = float(res['Dens'])
                maps['KB_IcingProb'][idx] = float(res['Ice'])
                maps['KB_Shear'][idx] = float(res['Shear'])
            except:
                continue

    df_target['KB_WindPotential'] = df_target.index.map(maps['KB_WindPotential']).fillna(0.0)
    df_target['KB_Turbulence'] = df_target.index.map(maps['KB_Turbulence']).fillna(0.0)
    df_target['KB_DensityScore'] = df_target.index.map(maps['KB_DensityScore']).fillna(1.0)
    df_target['KB_IcingProb'] = df_target.index.map(maps['KB_IcingProb']).fillna(0.0)
    df_target['KB_Shear'] = df_target.index.map(maps['KB_Shear']).fillna(0.0)

    return df_target