import axios from 'axios';

export interface PlungeAlertPayload {
  symbol: string;
  current_price: number;
  reference_price: number;
  reference_type: 'n_day_high' | 'prior_close';
  drop_pct: number;
  threshold_pct: number;
  triggered_at: string;
}

export interface PlungeSummarySymbol {
  symbol: string;
  count: number;
  max_drop_pct: number;
  latest_price: number;
  latest_triggered_at: string;
}

export interface PlungeSummaryPayload {
  window_hours: number;
  generated_at: string;
  total_events: number;
  symbols: PlungeSummarySymbol[];
  omitted_symbols: number;
}

type LogFn = (level: string, message: string, meta?: Record<string, unknown>) => void;

export async function sendDiscordAlert(event: PlungeAlertPayload, writeLog?: LogFn): Promise<void> {
  const webhookUrl = process.env.DISCORD_WEBHOOK;
  if (!webhookUrl) {
    writeLog?.('ERROR', 'DISCORD_WEBHOOK not configured, cannot send plunge alert');
    return;
  }

  const referenceLabel = event.reference_type === 'n_day_high' ? 'N-day high' : 'prior close';

  try {
    await axios.post(webhookUrl, {
      embeds: [
        {
          title: `📉 Plunge alert: ${event.symbol}`,
          color: 0xf87171,
          fields: [
            { name: 'Current price', value: String(event.current_price), inline: true },
            { name: `Drop vs ${referenceLabel}`, value: `${event.drop_pct}% (threshold ${event.threshold_pct}%)`, inline: true },
            { name: 'Reference price', value: String(event.reference_price), inline: true }
          ],
          timestamp: event.triggered_at
        }
      ]
    });
  } catch (error) {
    writeLog?.('ERROR', 'Failed to post Discord alert', { error: (error as Error).message });
  }
}

export async function sendDiscordSummary(summary: PlungeSummaryPayload, writeLog?: LogFn): Promise<void> {
  const webhookUrl = process.env.DISCORD_WEBHOOK;
  if (!webhookUrl) {
    writeLog?.('ERROR', 'DISCORD_WEBHOOK not configured, cannot send plunge summary');
    return;
  }

  const fields = summary.symbols.map((s) => ({
    name: s.symbol,
    value: `${s.count} alert${s.count === 1 ? '' : 's'}, max drop ${s.max_drop_pct}%, latest ${s.latest_price}`,
    inline: true
  }));

  try {
    await axios.post(webhookUrl, {
      embeds: [
        {
          title: `📊 Plunge summary (last ${summary.window_hours}h)`,
          color: 0xf59e0b,
          description: `${summary.total_events} alert(s) across ${summary.symbols.length + summary.omitted_symbols} symbol(s)`,
          fields,
          footer: summary.omitted_symbols > 0 ? { text: `+${summary.omitted_symbols} more not shown` } : undefined,
          timestamp: summary.generated_at
        }
      ]
    });
  } catch (error) {
    writeLog?.('ERROR', 'Failed to post Discord summary', { error: (error as Error).message });
  }
}
