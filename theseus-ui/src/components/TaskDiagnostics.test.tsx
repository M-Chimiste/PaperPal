import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import TaskDiagnostics from './TaskDiagnostics';

vi.mock('../services/api', () => ({ runtimeApi: {
  diagnostics: vi.fn().mockResolvedValue({ data: { events: [], deliveries: [{ status: 'uncertain', updated_at: '' }] } }),
  retry: vi.fn(), resolveDelivery: vi.fn(),
} }));

describe('TaskDiagnostics', () => {
  it('blocks retry while delivery is uncertain and offers explicit resolution', async () => {
    render(<QueryClientProvider client={new QueryClient()}><TaskDiagnostics taskId="test-task" failed onRetry={() => {}} /></QueryClientProvider>);
    fireEvent.click(screen.getByText('Diagnostics'));
    await screen.findByText('I confirmed delivery');
    expect(screen.getByText('Retry run')).toBeDisabled();
    expect(screen.getByText('I confirmed it was not sent')).toBeEnabled();
  });
});
