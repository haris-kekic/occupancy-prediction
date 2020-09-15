# Einfache Klasse ohne Funktionalität. Dient als Rückgabeobjekt
# mit zusätzlichen Metadaten der eigentlichen Daten
# Die Daten selbst werden als Pandas DataFrame Objekt hinterlegt.
class VisualData:
    def __init__(self, dataset_name, title, start_date, end_date, data, timestamp = '', time_axis_step = 60):
        self.dataset_name = dataset_name # Name des Gebäude-Datensatzes
        self.title = title # Meistens genutzt von Plotter für den Titel des Plots
        self.timestamp = timestamp # Zeitstempel für Anfangs und Enddatum als String
        self.start_date = start_date # Anfangsdatum nach dem gefiltert wurde
        self.end_date = end_date # Enddatum nach dem gefiltert wurde
        self.time_axis_step = time_axis_step # Schrittgröße der Zeitachse (z.B. wenn 60, alle 60 Zeitpunkte wird der Zeiteintrag genommen)
        self.data = data # Daten, die Zurückgeliefert werden
