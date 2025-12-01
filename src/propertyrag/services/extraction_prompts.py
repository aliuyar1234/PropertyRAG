"""Extraction prompts for different document types."""

MIETVERTRAG_PROMPT = """Extrahiere die folgenden Informationen aus diesem Mietvertrag.
Antworte im JSON-Format. Wenn ein Feld nicht gefunden wird, setze es auf null.

Zu extrahierende Felder:
- vermieter: Name und Adresse des Vermieters
- mieter: Name und Adresse des Mieters
- objekt_adresse: Vollständige Adresse des Mietobjekts
- objekt_typ: Art des Objekts (z.B. "Wohnung", "Gewerbe", "Büro", "Lager")
- flaeche_qm: Fläche in Quadratmetern (nur Zahl)
- nettomiete_eur: Kaltmiete/Nettomiete in Euro (nur Zahl)
- nebenkosten_eur: Nebenkostenvorauszahlung in Euro (nur Zahl)
- bruttomiete_eur: Gesamtmiete/Warmmiete in Euro (nur Zahl)
- mietbeginn: Mietbeginn im Format YYYY-MM-DD
- mietende: Mietende im Format YYYY-MM-DD (falls befristet)
- befristet: true wenn befristet, false wenn unbefristet
- kuendigungsfrist_monate: Kündigungsfrist in Monaten (nur Zahl)
- indexierung: Beschreibung der Indexklausel falls vorhanden
- kaution_eur: Kautionshöhe in Euro (nur Zahl)
- sondervereinbarungen: Liste von besonderen Vereinbarungen

Dokumenttext:
{text}"""

GUTACHTEN_PROMPT = """Extrahiere die folgenden Informationen aus diesem Immobiliengutachten.
Antworte im JSON-Format. Wenn ein Feld nicht gefunden wird, setze es auf null.

Zu extrahierende Felder:
- gutachter: Name des Gutachters/Sachverständigen
- bewertungsstichtag: Bewertungsstichtag im Format YYYY-MM-DD
- verkehrswert_eur: Verkehrswert in Euro (nur Zahl)
- ertragswert_eur: Ertragswert in Euro (nur Zahl, falls genannt)
- sachwert_eur: Sachwert in Euro (nur Zahl, falls genannt)
- nutzungsart: Art der Nutzung (z.B. "Wohnnutzung", "Gewerbenutzung", "Mischnutzung")
- baujahr: Baujahr des Gebäudes (nur Zahl)
- wohnflaeche_qm: Wohnfläche in Quadratmetern (nur Zahl)
- grundstuecksflaeche_qm: Grundstücksfläche in Quadratmetern (nur Zahl)
- adresse: Vollständige Adresse der bewerteten Immobilie

Dokumenttext:
{text}"""

GRUNDBUCHAUSZUG_PROMPT = """Extrahiere die folgenden Informationen aus diesem Grundbuchauszug.
Antworte im JSON-Format. Wenn ein Feld nicht gefunden wird, setze es auf null.

Zu extrahierende Felder:
- grundbuchamt: Zuständiges Grundbuchamt
- blatt_nummer: Grundbuchblattnummer
- flurnummer: Flurnummer(n)
- gemarkung: Gemarkung
- grundstuecksgroesse_qm: Grundstücksgröße in Quadratmetern (nur Zahl)
- eigentuemer: Liste der Eigentümer (Namen)
- belastungen: Liste der Belastungen, jeweils mit:
  - typ: Art der Belastung (z.B. "Grundschuld", "Hypothek", "Dienstbarkeit", "Wegerecht")
  - betrag_eur: Betrag in Euro falls vorhanden (nur Zahl)
  - glaeubiger: Gläubiger falls angegeben
  - beschreibung: Nähere Beschreibung
- stand_datum: Stand/Datum des Auszugs im Format YYYY-MM-DD

Dokumenttext:
{text}"""

NEBENKOSTENABRECHNUNG_PROMPT = """Extrahiere die folgenden Informationen aus dieser Nebenkostenabrechnung.
Antworte im JSON-Format. Wenn ein Feld nicht gefunden wird, setze es auf null.

Zu extrahierende Felder:
- abrechnungszeitraum_von: Beginn des Abrechnungszeitraums im Format YYYY-MM-DD
- abrechnungszeitraum_bis: Ende des Abrechnungszeitraums im Format YYYY-MM-DD
- objekt_adresse: Adresse des Objekts
- mieter: Name des Mieters
- gesamtkosten_eur: Gesamte Nebenkosten in Euro (nur Zahl)
- vorauszahlungen_eur: Geleistete Vorauszahlungen in Euro (nur Zahl)
- nachzahlung_eur: Nachzahlungsbetrag in Euro (nur Zahl, falls Nachzahlung)
- guthaben_eur: Guthabenbetrag in Euro (nur Zahl, falls Guthaben)
- positionen: Liste der Kostenpositionen, jeweils mit:
  - bezeichnung: Name der Kostenposition (z.B. "Heizkosten", "Wasser", "Müllabfuhr")
  - betrag_eur: Anteil des Mieters in Euro (nur Zahl)
  - umlageschluessel: Verteilungsschlüssel (z.B. "nach qm", "nach Personen", "Verbrauch")

Dokumenttext:
{text}"""

# Mapping of document types to prompts
EXTRACTION_PROMPTS = {
    "mietvertrag": MIETVERTRAG_PROMPT,
    "gutachten": GUTACHTEN_PROMPT,
    "grundbuchauszug": GRUNDBUCHAUSZUG_PROMPT,
    "nebenkostenabrechnung": NEBENKOSTENABRECHNUNG_PROMPT,
}
