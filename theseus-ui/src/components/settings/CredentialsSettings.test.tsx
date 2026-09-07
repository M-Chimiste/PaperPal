import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SnackbarProvider } from '../../contexts/SnackbarContext';
import { CredentialsSettings } from './CredentialsSettings';

const update = vi.hoisted(() => vi.fn().mockResolvedValue({ data: { status: 'success' } }));
vi.mock('../../services/api', () => ({ settingsApi: {
  getCredentials: vi.fn().mockResolvedValue({ data: { OPENAI_API_KEY: { configured: true, value: '' }, OLLAMA_URL: { configured: true, value: 'http://localhost:11434' } } }),
  updateCredentials: update,
} }));

describe('CredentialsSettings', () => {
  it('keeps configured secrets blank and submits replacements', async () => {
    render(<QueryClientProvider client={new QueryClient()}><SnackbarProvider><CredentialsSettings /></SnackbarProvider></QueryClientProvider>);
    fireEvent.click(screen.getByText('API Credentials'));
    const field = await screen.findByLabelText('OPENAI_API_KEY');
    await waitFor(() => expect(field).toHaveAttribute('placeholder', 'Configured — enter a replacement'));
    expect(field).toHaveValue('');
    fireEvent.change(field, { target: { value: 'replacement' } });
    fireEvent.click(screen.getByText('Apply Credentials'));
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({ OPENAI_API_KEY: 'replacement' })));
    await waitFor(() => expect(field).toHaveValue(''));
  });
});
