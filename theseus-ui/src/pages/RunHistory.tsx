import TaskDiagnostics from '../components/TaskDiagnostics';
import React, { useState, useEffect, useCallback } from 'react';
import { 
    Box, 
    Typography, 
    CircularProgress, 
    Alert, 
    Table, 
    TableBody, 
    TableCell, 
    TableContainer, 
    TableHead, 
    TableRow, 
    Paper, 
    TablePagination, 
    Grid,
    Button,
    Chip,
    IconButton,
    Tooltip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogContentText,
    DialogActions,
    Snackbar,
    Collapse
} from '@mui/material';
import StopIcon from '@mui/icons-material/Stop';
import FilterListIcon from '@mui/icons-material/FilterList';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { DatePicker, LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { format } from 'date-fns'; // Correct import for date-fns v3
import { getTaskHistory, settingsApi } from '../services/api';
import type { TaskHistoryEntry } from '../services/api';
import { useLayout } from '../contexts/LayoutContext';

const RunHistory: React.FC = () => {
    const { headerHeight } = useLayout(); // Get dynamic header height
    const [tasks, setTasks] = useState<TaskHistoryEntry[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(0);
    const [rowsPerPage] = useState(10);
    const [fromDate, setFromDate] = useState<Date | null>(null);
    const [toDate, setToDate] = useState<Date | null>(null);
    const [abortDialogOpen, setAbortDialogOpen] = useState(false);
    const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
    const [snackbarMessage, setSnackbarMessage] = useState<string | null>(null);
    const [aborting, setAborting] = useState<string | null>(null); // Track which task is being aborted
    const [showDateFilter, setShowDateFilter] = useState(false); // New state for date filter toggle

    const fetchTasks = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const fromDateStr = fromDate ? format(fromDate, 'yyyy-MM-dd') : undefined;
            const toDateStr = toDate ? format(toDate, 'yyyy-MM-dd') : undefined;
            const fetchedTasks = await getTaskHistory(100, fromDateStr, toDateStr);
            setTasks(fetchedTasks);
        } catch (err) {
            console.error("Error fetching task history:", err);
            setError('Failed to fetch run history. Please try again.');
        }
        setLoading(false);
    }, [fromDate, toDate]);

    useEffect(() => {
        fetchTasks();
    }, [fetchTasks]);

    const handleChangePage = (_event: unknown, newPage: number) => {
        setPage(newPage);
    };

    const handleAbortClick = (taskId: string) => {
        setSelectedTaskId(taskId);
        setAbortDialogOpen(true);
    };

    const handleAbortConfirm = async () => {
        if (selectedTaskId) {
            setAborting(selectedTaskId);
            try {
                await settingsApi.abortTask(selectedTaskId);
                setSnackbarMessage('Task aborted successfully');
                // Refresh the task list
                fetchTasks();
            } catch (err) {
                console.error("Error aborting task:", err);
                setSnackbarMessage('Error aborting task');
            } finally {
                setAborting(null);
            }
        }
        setAbortDialogOpen(false);
        setSelectedTaskId(null);
    };

    const handleAbortCancel = () => {
        setAbortDialogOpen(false);
        setSelectedTaskId(null);
    };

    const getStatusChip = (status: string) => {
        const statusLower = status.toLowerCase();
        if (statusLower === 'completed') {
            return <Chip label="Completed" color="success" size="small" />;
        } else if (statusLower === 'failed') {
            return <Chip label="Failed" color="error" size="small" />;
        } else if (statusLower === 'processing' || statusLower === 'pending') {
            return <Chip label="In Progress" color="warning" size="small" />;
        } else {
            return <Chip label={status} color="default" size="small" />;
        }
    };

    const isTaskInProgress = (status: string) => {
        const statusLower = status.toLowerCase();
        return statusLower === 'processing' || statusLower === 'pending';
    };

    const formatTaskType = (taskType: string) => {
        // Capitalize and format task type nicely
        return taskType.charAt(0).toUpperCase() + taskType.slice(1).replace(/_/g, ' ');
    };

    return (
        <LocalizationProvider dateAdapter={AdapterDateFns}>
            <Box sx={{ pt: `${headerHeight + 24}px`, pb: 3, px: 3 }}>
                <Typography variant="h4" gutterBottom component="div" sx={{ mb: 3 }}>
                    Run History
                </Typography>

                <Paper sx={{ mb: 2, p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: showDateFilter ? 2 : 0 }}>
                        <Button
                            startIcon={<FilterListIcon />}
                            endIcon={showDateFilter ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                            onClick={() => setShowDateFilter(!showDateFilter)}
                            variant="outlined"
                            size="small"
                        >
                            Date Filter
                        </Button>
                    </Box>
                    
                    <Collapse in={showDateFilter}>
                        <Grid container spacing={2} alignItems="center" justifyContent="flex-start">
                            <Grid size={{ xs: 12, sm: 4, md: 3 }}>
                                <DatePicker
                                    label="From Date"
                                    value={fromDate}
                                    onChange={(newValue) => {
                                        setFromDate(newValue);
                                        setPage(0); // Reset page when date changes
                                    }}
                                    slotProps={{ textField: { fullWidth: true, size: 'small' } }}
                                />
                            </Grid>
                            <Grid size={{ xs: 12, sm: 4, md: 3 }}>
                                <DatePicker
                                    label="To Date"
                                    value={toDate}
                                    onChange={(newValue) => {
                                        setToDate(newValue);
                                        setPage(0); // Reset page when date changes
                                    }}
                                    slotProps={{ textField: { fullWidth: true, size: 'small' } }}
                                    minDate={fromDate || undefined} // Prevent selecting toDate before fromDate
                                />
                            </Grid>
                            {(fromDate || toDate) && (
                                <Grid size={{ xs: 12, sm: 4, md: 3 }}>
                                    <Button
                                        variant="text"
                                        size="small"
                                        onClick={() => {
                                            setFromDate(null);
                                            setToDate(null);
                                            setPage(0);
                                        }}
                                    >
                                        Clear Filters
                                    </Button>
                                </Grid>
                            )}
                        </Grid>
                    </Collapse>
                </Paper>

                {loading && <CircularProgress sx={{ display: 'block', margin: 'auto', mt: 3 }} />}
                {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
                {!loading && !error && (
                    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
                        <TableContainer sx={{ maxHeight: 600 }}>
                            <Table stickyHeader aria-label="run history table">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Task ID</TableCell>
                                        <TableCell align="center">Status</TableCell>
                                        <TableCell>Start Time</TableCell>
                                        <TableCell>Task Type</TableCell>
                                        <TableCell align="center">Actions</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {tasks.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((task) => (
                                        <TableRow hover role="checkbox" tabIndex={-1} key={task.task_id + task.start_time}>
                                            <TableCell>{task.task_id}</TableCell>
                                            <TableCell align="center">{getStatusChip(task.status)}</TableCell>
                                            <TableCell>{task.start_time}</TableCell>
                                            <TableCell>{formatTaskType(task.task_type)}</TableCell>
                                            <TableCell align="center">
                                                <TaskDiagnostics taskId={task.task_id} failed={task.status.toLowerCase() === "failed"} onRetry={() => { void fetchTasks(); }} />
                                                {isTaskInProgress(task.status) && (
                                                    <Tooltip title="Abort Task">
                                                        <IconButton 
                                                            color="error" 
                                                            size="small" 
                                                            onClick={() => handleAbortClick(task.task_id)}
                                                            disabled={aborting === task.task_id}
                                                        >
                                                            {aborting === task.task_id ? (
                                                                <CircularProgress size={16} />
                                                            ) : (
                                                                <StopIcon />
                                                            )}
                                                        </IconButton>
                                                    </Tooltip>
                                                )}
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                    {tasks.length === 0 && (
                                        <TableRow>
                                            <TableCell colSpan={5} align="center">
                                                No task history found for the selected criteria.
                                            </TableCell>
                                        </TableRow>
                                    )}
                                </TableBody>
                            </Table>
                        </TableContainer>
                        <TablePagination
                            rowsPerPageOptions={[]}
                            component="div"
                            count={tasks.length}
                            rowsPerPage={rowsPerPage}
                            page={page}
                            onPageChange={handleChangePage}
                        />
                    </Paper>
                )}
            </Box>
            <Snackbar
                open={snackbarMessage !== null}
                autoHideDuration={6000}
                onClose={() => setSnackbarMessage(null)}
                message={snackbarMessage}
            />
            <Dialog
                open={abortDialogOpen}
                onClose={handleAbortCancel}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">{"Abort Task"}</DialogTitle>
                <DialogContent>
                    <DialogContentText id="alert-dialog-description">
                        Are you sure you want to abort this task?
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleAbortCancel}>Cancel</Button>
                    <Button onClick={handleAbortConfirm} autoFocus>
                        Abort
                    </Button>
                </DialogActions>
            </Dialog>
        </LocalizationProvider>
    );
};

export default RunHistory;
