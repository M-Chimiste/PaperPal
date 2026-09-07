import React, { useEffect, useState } from 'react';
import {
  Accordion, AccordionDetails, AccordionSummary, Box, Button,
  IconButton, TextField, Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { settingsApi } from '../../services/api';
import { useSnackbar } from '../../contexts/SnackbarContext';

const CREDENTIAL_KEYS = [
  'GOOGLE_API_KEY',
  'ANTHROPIC_API_KEY',
  'OPENAI_API_KEY',
  'GMAIL_SENDER_ADDRESS',
  'GMAIL_APP_PASSWORD',
  'OLLAMA_URL',
  'CLIENT_ID',
  'PROJECT_ID',
  'CLIENT_SECRET',
  'CUSTOM_OAI_BASE_URL',
  'CUSTOM_OAI_API_KEY',
  'KAGGLE_USERNAME',
  'KAGGLE_KEY',
];

// Plain-text (non-secret) keys that don't get the show/hide toggle.
const PLAIN_KEYS = new Set(['OLLAMA_URL', 'CUSTOM_OAI_BASE_URL']);

/** API credential management (extracted from Settings.tsx in F3). */
export const CredentialsSettings: React.FC = () => {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useSnackbar();
  const [credValues, setCredValues] = useState<Record<string, string>>({});
  const [showCreds, setShowCreds] = useState<Record<string, boolean>>({});

  const { data: credentials } = useQuery({
    queryKey: ['credentials'],
    queryFn: () => settingsApi.getCredentials().then(res => res.data),
  });

  useEffect(() => {
    if (credentials) {
      setCredValues(Object.fromEntries(Object.entries(credentials).map(([key, status]) => [key, status.value])));
    }
  }, [credentials]);

  const updateCredentialsMutation = useMutation({
    mutationFn: (data: Record<string, string>) => settingsApi.updateCredentials(data),
    onSuccess: () => {
      setCredValues({});
      queryClient.invalidateQueries({ queryKey: ['credentials'] });
      showSuccess('Credentials updated');
    },
    onError: (error: Error) => showError(error.message),
  });

  return (
    <Accordion sx={{ mb: 4 }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="h5" fontWeight={600}>
          API Credentials
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 700 }}>
          {CREDENTIAL_KEYS.map((key) => (
            <TextField
              key={key}
              label={key}
              type={PLAIN_KEYS.has(key) ? 'text' : (showCreds[key] ? 'text' : 'password')}
              value={credValues[key] || ''}
              placeholder={credentials?.[key]?.configured ? 'Configured — enter a replacement' : 'Not configured'}
              helperText="Leave blank to keep the existing value"
              onChange={e => setCredValues({ ...credValues, [key]: e.target.value })}
              InputProps={{
                endAdornment:
                  PLAIN_KEYS.has(key) ? null : (
                    <IconButton onClick={() => setShowCreds({ ...showCreds, [key]: !showCreds[key] })}>
                      {showCreds[key] ? <VisibilityOffIcon /> : <VisibilityIcon />}
                    </IconButton>
                  )
              }}
            />
          ))}
          <Box sx={{ mt: 2 }}>
            <Button
              variant="contained"
              onClick={() => updateCredentialsMutation.mutate(credValues)}
              disabled={updateCredentialsMutation.isPending}
            >
              Apply Credentials
            </Button>
          </Box>
        </Box>
      </AccordionDetails>
    </Accordion>
  );
};
