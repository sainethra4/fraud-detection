"""Create a plain-English PDF summary of the model evaluation."""
from __future__ import annotations

from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def _currency(value: float) -> str:
    return f"AUD ${value:,.2f}"


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E1EF"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 10 * mm, "FraudShield - Auto-insurance fraud detection")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_performance_pdf(report: dict) -> bytes:
    """Return a polished, static PDF suitable for non-technical readers."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=22 * mm,
        title="FraudShield Model Performance Report",
        author="FraudShield",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=24, leading=29, textColor=colors.HexColor("#0F172A"), alignment=TA_LEFT)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10, leading=15,
                              textColor=colors.HexColor("#64748B"))
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold",
                             fontSize=14, leading=19, textColor=colors.HexColor("#1E3A5F"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=15, textColor=colors.HexColor("#334155"))
    note = ParagraphStyle("Note", parent=body, fontSize=9, leading=13, textColor=colors.HexColor("#475569"), backColor=colors.HexColor("#EFF6FF"), borderColor=colors.HexColor("#BFDBFE"), borderWidth=.5, borderPadding=8)

    impact = report["financial_impact_aud"]
    matrix = report["confusion_matrix"]
    story = [
        Paragraph("FraudShield", title),
        Paragraph("Auto-insurance fraud detection - model performance report", subtitle),
        Paragraph(f"Prepared: {date.today().isoformat()} | Dataset: {report['data'].get('dataset', 'insurance claims')}", subtitle),
        Spacer(1, 7), HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB")), Spacer(1, 10),
        Paragraph("Executive summary", heading),
        Paragraph(
            "This report explains how well the XGBoost model identifies claims that should be prioritised for human fraud investigation. "
            "A model score is not proof of fraud and must not be used as the sole basis for an adverse claims decision.", body),
        Spacer(1, 8),
    ]
    metric_rows = [
        ["Measure", "Result", "Plain-English meaning"],
        ["Accuracy", f"{report['accuracy']:.2%}", "Overall share of correct fraud/non-fraud classifications."],
        ["Precision", f"{report['precision']:.2%}", "Of claims selected for review, this share was confirmed fraud in testing."],
        ["Sensitivity (recall)", f"{report['sensitivity_recall']:.2%}", "Share of known fraudulent claims detected by the model."],
        ["Specificity", f"{report['specificity']:.2%}", "Share of genuine claims correctly left unflagged."],
        ["ROC-AUC", f"{report['roc_auc']:.4f}", "Ability to rank fraudulent claims above genuine claims; 0.5 is random and 1.0 is perfect."],
        ["Review threshold", f"{report['threshold']:.2f}", "Claims at or above this probability are recommended for investigation."],
    ]
    metric_table = Table(metric_rows, colWidths=[36 * mm, 30 * mm, 98 * mm], repeatRows=1)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story += [Paragraph("Performance results", heading), metric_table, Spacer(1, 9)]
    story += [Paragraph("Classification results", heading)]
    confusion = Table([
        ["", "Model: not flagged", "Model: investigate"],
        ["Actual genuine claim", str(matrix["true_negative"]), str(matrix["false_positive"])],
        ["Actual fraudulent claim", str(matrix["false_negative"]), str(matrix["true_positive"])],
    ], colWidths=[60 * mm, 52 * mm, 52 * mm])
    confusion.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0E7FF")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [confusion, Spacer(1, 9), Paragraph("Financial impact assessment", heading)]
    finance = Table([
        ["Measure", "Value"],
        ["Test claim value", _currency(impact["test_claim_value"])],
        ["Confirmed fraud value", _currency(impact["confirmed_fraud_value"])],
        ["Potential fraud value prioritised", _currency(impact["potential_fraud_value_prioritised"])],
    ], colWidths=[104 * mm, 60 * mm])
    finance.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [finance, Spacer(1, 8), Paragraph(impact["note"], note)]
    recommendations = [
        "Prioritise high-probability claims for investigator review, while displaying claim value and supporting evidence.",
        "Review the threshold regularly against investigation capacity, false-positive impact, and fraud missed.",
        "Track investigation outcomes, recovery value, customer complaints, and model drift before retraining.",
        "Keep a human investigator responsible for every final claims decision and record the supporting evidence.",
    ]
    closing = [Paragraph("Recommended next actions", heading)]
    closing.extend(Paragraph(f"- {item}", body) for item in recommendations)
    closing += [Spacer(1, 8), Paragraph("Governance note", heading), Paragraph(
        "Before production use, complete privacy, security, retention, access/correction, fairness, audit, and legal review. "
        "Refer to the project compliance checklist for the operational controls and escalation requirements.", body)]
    story.append(KeepTogether(closing))
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
