"""
Appointment Service — database operations for the agent swarm.
Agents call these functions to read/write the appointments table.
"""

from datetime import datetime, timedelta
from api.services.supabase_client import get_supabase_admin, log_audit_event
import json

# Default appointment durations by type (minutes)
APPOINTMENT_DURATIONS = {
    "cleaning": 60,
    "exam": 30,
    "filling": 45,
    "crown": 90,
    "root_canal": 90,
    "extraction": 60,
    "whitening": 60,
    "consultation": 30,
    "emergency": 30,
    "follow_up": 15,
    "general": 30,
}

# Business hours
BUSINESS_HOURS = {
    "start": 8,  # 8 AM
    "end": 17,   # 5 PM
    "slot_minutes": 30,
    "days": [0, 1, 2, 3, 4],  # Mon-Fri
}


async def get_appointments_for_date(
    workspace_id: str,
    date: str,  # "2026-02-24"
) -> list[dict]:
    """Get all appointments for a specific date."""
    supabase = get_supabase_admin()

    start = f"{date}T00:00:00Z"
    end = f"{date}T23:59:59Z"

    result = (
        supabase.table("appointments")
        .select("id, title, appointment_type, start_time, end_time, duration_minutes, status, patient_id")
        .eq("workspace_id", workspace_id)
        .gte("start_time", start)
        .lte("start_time", end)
        .neq("status", "cancelled")
        .order("start_time")
        .execute()
    )

    return result.data or []


async def get_appointments_for_range(
    workspace_id: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Get appointments for a date range."""
    supabase = get_supabase_admin()

    result = (
        supabase.table("appointments")
        .select("id, title, appointment_type, start_time, end_time, duration_minutes, status, patient_id")
        .eq("workspace_id", workspace_id)
        .gte("start_time", f"{start_date}T00:00:00Z")
        .lte("start_time", f"{end_date}T23:59:59Z")
        .neq("status", "cancelled")
        .order("start_time")
        .execute()
    )

    return result.data or []


async def get_patient_appointments(
    workspace_id: str,
    patient_id: str,
    upcoming_only: bool = True,
) -> list[dict]:
    """Get appointments for a specific patient."""
    supabase = get_supabase_admin()

    query = (
        supabase.table("appointments")
        .select("id, title, appointment_type, start_time, end_time, duration_minutes, status")
        .eq("workspace_id", workspace_id)
        .eq("patient_id", patient_id)
        .neq("status", "cancelled")
        .order("start_time")
    )

    if upcoming_only:
        query = query.gte("start_time", datetime.utcnow().isoformat())

    result = query.execute()
    return result.data or []


async def check_availability(
    workspace_id: str,
    date: str,
    duration_minutes: int = 30,
) -> list[dict]:
    """
    Find available time slots for a given date and duration.
    Returns list of available slots: [{"start": "09:00", "end": "09:30"}, ...]
    """
    existing = await get_appointments_for_date(workspace_id, date)

    # Parse date to check if it's a business day
    check_date = datetime.strptime(date, "%Y-%m-%d")
    if check_date.weekday() not in BUSINESS_HOURS["days"]:
        return []  # Not a business day

    # Build list of busy periods
    busy = []
    for appt in existing:
        start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
        busy.append((start.hour * 60 + start.minute, end.hour * 60 + end.minute))

    # Find free slots
    available = []
    slot_start = BUSINESS_HOURS["start"] * 60  # 8:00 = 480

    while slot_start + duration_minutes <= BUSINESS_HOURS["end"] * 60:
        slot_end = slot_start + duration_minutes

        # Check if slot overlaps with any busy period
        is_free = True
        for busy_start, busy_end in busy:
            if slot_start < busy_end and slot_end > busy_start:
                is_free = False
                break

        if is_free:
            start_h, start_m = divmod(slot_start, 60)
            end_h, end_m = divmod(slot_end, 60)
            available.append({
                "start": f"{start_h:02d}:{start_m:02d}",
                "end": f"{end_h:02d}:{end_m:02d}",
            })

        slot_start += BUSINESS_HOURS["slot_minutes"]

    return available


async def find_next_available(
    workspace_id: str,
    duration_minutes: int = 30,
    days_ahead: int = 14,
    max_results: int = 5,
) -> list[dict]:
    """
    Find the next available slots across multiple days.
    Returns: [{"date": "2026-02-25", "slots": [{"start": "09:00", "end": "09:30"}, ...]}]
    """
    results = []
    today = datetime.utcnow().date()

    for day_offset in range(days_ahead):
        check_date = today + timedelta(days=day_offset)

        # Skip weekends
        if check_date.weekday() not in BUSINESS_HOURS["days"]:
            continue

        # Skip today if past business hours
        if day_offset == 0:
            now = datetime.utcnow()
            if now.hour >= BUSINESS_HOURS["end"]:
                continue

        date_str = check_date.isoformat()
        slots = await check_availability(workspace_id, date_str, duration_minutes)

        # Filter out past slots for today
        if day_offset == 0:
            now = datetime.utcnow()
            current_minutes = now.hour * 60 + now.minute
            slots = [s for s in slots if int(s["start"].split(":")[0]) * 60 + int(s["start"].split(":")[1]) > current_minutes]

        if slots:
            results.append({
                "date": date_str,
                "day_name": check_date.strftime("%A"),
                "slots": slots[:3],  # Top 3 per day
            })

        if len(results) >= max_results:
            break

    return results


async def book_appointment(
    workspace_id: str,
    date: str,
    time: str,  # "09:00"
    appointment_type: str = "general",
    patient_id: str | None = None,
    title: str | None = None,
    notes: str | None = None,
    source: str = "concierge",
) -> dict:
    """Book a new appointment."""
    supabase = get_supabase_admin()

    duration = APPOINTMENT_DURATIONS.get(appointment_type, 30)
    start_time = datetime.fromisoformat(f"{date}T{time}:00")
    end_time = start_time + timedelta(minutes=duration)

    # Verify slot is available
    slots = await check_availability(workspace_id, date, duration)
    is_available = any(s["start"] == time for s in slots)

    if not is_available:
        return {
            "success": False,
            "error": "This time slot is not available",
            "available_slots": slots[:5],
        }

    result = (
        supabase.table("appointments")
        .insert({
            "workspace_id": workspace_id,
            "patient_id": patient_id,
            "title": title or appointment_type.replace("_", " ").title(),
            "appointment_type": appointment_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration,
            "notes": notes,
            "source": source,
            "status": "scheduled",
        })
        .execute()
    )

    if result.data and len(result.data) > 0:
        result_data = result.data[0] if isinstance(result.data, list) else result_data
        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="concierge",
            action="appointment_booked",
            resource_type="appointment",
            resource_id=result_data["id"],
            metadata={
                "date": date,
                "time": time,
                "type": appointment_type,
                "patient_id": patient_id,
                "source": source,
            },
        )

        return {
            "success": True,
            "appointment": {
                "id": result.data["id"],
                "date": date,
                "time": time,
                "type": appointment_type,
                "duration": duration,
                "status": "scheduled",
            },
        }

    return {"success": False, "error": "Failed to create appointment"}


async def cancel_appointment(
    workspace_id: str,
    appointment_id: str | None = None,
    patient_id: str | None = None,
    reason: str = "Patient requested cancellation",
) -> dict:
    """
    Cancel an appointment by ID or by patient's next upcoming appointment.
    """
    supabase = get_supabase_admin()

    # Find the appointment
    if appointment_id:
        result = (
            supabase.table("appointments")
            .select("*")
            .eq("id", appointment_id)
            .eq("workspace_id", workspace_id)
            .single()
            .execute()
        )
    elif patient_id:
        # Cancel next upcoming appointment for this patient
        result = (
            supabase.table("appointments")
            .select("*")
            .eq("workspace_id", workspace_id)
            .eq("patient_id", patient_id)
            .gte("start_time", datetime.utcnow().isoformat())
            .neq("status", "cancelled")
            .order("start_time")
            .limit(1)
            .single()
            .execute()
        )
    else:
        return {"success": False, "error": "Need appointment_id or patient_id"}

    if not result.data:
        return {"success": False, "error": "Appointment not found"}

    appt = result.data

    # Cancel it
    supabase.table("appointments").update({
        "status": "cancelled",
        "cancellation_reason": reason,
    }).eq("id", appt["id"]).execute()

    await log_audit_event(
        workspace_id=workspace_id,
        actor_type="agent",
        actor_id="concierge",
        action="appointment_cancelled",
        resource_type="appointment",
        resource_id=appt["id"],
        metadata={
            "reason": reason,
            "original_date": appt["start_time"],
            "type": appt["appointment_type"],
        },
    )

    # Find next available slots to offer
    duration = appt.get("duration_minutes", 30)
    next_slots = await find_next_available(workspace_id, duration, days_ahead=14, max_results=3)

    return {
        "success": True,
        "cancelled_appointment": {
            "id": appt["id"],
            "date": appt["start_time"][:10],
            "time": appt["start_time"][11:16],
            "type": appt["appointment_type"],
        },
        "suggested_reschedule": next_slots,
    }


async def reschedule_appointment(
    workspace_id: str,
    appointment_id: str,
    new_date: str,
    new_time: str,
) -> dict:
    """Reschedule an existing appointment to a new date/time."""
    supabase = get_supabase_admin()

    # Get current appointment
    current = (
        supabase.table("appointments")
        .select("*")
        .eq("id", appointment_id)
        .eq("workspace_id", workspace_id)
        .single()
        .execute()
    )

    if not current.data:
        return {"success": False, "error": "Appointment not found"}

    appt = current.data
    duration = appt.get("duration_minutes", 30)

    # Check if new slot is available
    slots = await check_availability(workspace_id, new_date, duration)
    is_available = any(s["start"] == new_time for s in slots)

    if not is_available:
        return {
            "success": False,
            "error": "The requested time is not available",
            "available_slots": slots[:5],
        }

    # Update
    new_start = datetime.fromisoformat(f"{new_date}T{new_time}:00")
    new_end = new_start + timedelta(minutes=duration)

    supabase.table("appointments").update({
        "start_time": new_start.isoformat(),
        "end_time": new_end.isoformat(),
        "status": "scheduled",
    }).eq("id", appointment_id).execute()

    await log_audit_event(
        workspace_id=workspace_id,
        actor_type="agent",
        actor_id="concierge",
        action="appointment_rescheduled",
        resource_type="appointment",
        resource_id=appointment_id,
        metadata={
            "old_date": appt["start_time"][:10],
            "old_time": appt["start_time"][11:16],
            "new_date": new_date,
            "new_time": new_time,
        },
    )

    return {
        "success": True,
        "rescheduled": {
            "id": appointment_id,
            "old_date": appt["start_time"][:10],
            "old_time": appt["start_time"][11:16],
            "new_date": new_date,
            "new_time": new_time,
            "type": appt["appointment_type"],
            "duration": duration,
        },
    }
