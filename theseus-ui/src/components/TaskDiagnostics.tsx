import { useState } from 'react';
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, Typography } from '@mui/material';
import { useQuery, useMutation } from '@tanstack/react-query';
import { runtimeApi } from '../services/api';

export default function TaskDiagnostics({ taskId, failed, onRetry }: { taskId: string; failed: boolean; onRetry: () => void }) {
  const [open, setOpen] = useState(false);
  const diagnostics = useQuery({ queryKey: ['diagnostics', taskId], queryFn: () => runtimeApi.diagnostics(taskId).then(r => r.data), enabled: open });
  const retry = useMutation({ mutationFn: () => runtimeApi.retry(taskId), onSuccess: () => { setOpen(false); onRetry(); } });
  const resolve = useMutation({ mutationFn: (resolution: 'sent' | 'retry') => runtimeApi.resolveDelivery(taskId, resolution), onSuccess: () => { void diagnostics.refetch(); } });
  const uncertain = diagnostics.data?.deliveries.some(d => ['sending', 'uncertain'].includes(d.status));
  return <>
    <Button size="small" onClick={() => setOpen(true)}>Diagnostics</Button>
    <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="md">
      <DialogTitle>Run diagnostics</DialogTitle>
      <DialogContent>
        <Stack spacing={2}>
          {(diagnostics.error || retry.error || resolve.error) && <Alert severity="error">{(diagnostics.error || retry.error || resolve.error)?.message}</Alert>}
          {diagnostics.data?.task?.error && <Alert severity="error">{diagnostics.data.task.error}</Alert>}
          {diagnostics.isLoading && <Typography>Loading diagnostics…</Typography>}
          {uncertain && <Alert severity="warning">
            Email may already have been delivered. Check your sent mail before retrying.
            {failed && <Stack direction="row" spacing={1}>
              <Button disabled={resolve.isPending} onClick={() => resolve.mutate('sent')}>I confirmed delivery</Button>
              <Button disabled={resolve.isPending} onClick={() => resolve.mutate('retry')}>I confirmed it was not sent</Button>
            </Stack>}
          </Alert>}
          {diagnostics.data?.events.length === 0 && <Typography>No stage diagnostics recorded for this run.</Typography>}
          {diagnostics.data?.events.map((event, index) => <Stack key={`${event.created_at}-${index}`} direction="row" spacing={2}>
            <Typography sx={{ minWidth: 170 }}>{event.stage}</Typography>
            <Typography color={event.status === 'failed' ? 'error' : 'text.secondary'}>{event.status}</Typography>
            <Typography>{event.duration_ms === null ? '' : `${(event.duration_ms / 1000).toFixed(1)}s`}</Typography>
            <Typography>{event.error_type}</Typography>
          </Stack>)}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setOpen(false)}>Close</Button>
        {failed && <Button variant="contained" disabled={retry.isPending || uncertain || diagnostics.isLoading || !!diagnostics.error} onClick={() => retry.mutate()}>Retry run</Button>}
      </DialogActions>
    </Dialog>
  </>;
}
