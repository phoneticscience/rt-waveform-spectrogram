'''Tests for rt_wav_sgram module.

Run with: pytest tests/test_rt_wav_sgram.py -v
'''
import sys
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, 'src')

from rt_wav_sgram import RingBuffer


class TestRingBuffer:
    '''Tests for the RingBuffer class.'''

    def test_init(self):
        '''Test buffer initialization.'''
        rb = RingBuffer(1000)
        assert rb.size == 1000
        assert rb.write_idx == 0
        assert rb.buf.shape == (1000,)
        assert rb.buf.dtype == np.float32

    def test_write_simple(self):
        '''Test writing data smaller than buffer.'''
        rb = RingBuffer(100)
        data = np.arange(10, dtype=np.float32)
        rb.write(data)
        assert rb.write_idx == 10
        np.testing.assert_array_equal(rb.buf[:10], data)

    def test_write_wrap_around(self):
        '''Test writing data that wraps around buffer.'''
        rb = RingBuffer(100)
        # Write 90 samples
        data1 = np.ones(90, dtype=np.float32)
        rb.write(data1)
        assert rb.write_idx == 90

        # Write 20 more - should wrap around
        data2 = np.ones(20, dtype=np.float32) * 2
        rb.write(data2)
        assert rb.write_idx == 10  # (90 + 20) % 100

    def test_write_larger_than_buffer(self):
        '''Test writing data larger than buffer size.'''
        rb = RingBuffer(100)
        data = np.arange(150, dtype=np.float32)
        rb.write(data)
        # Should keep only last 100 samples
        assert rb.write_idx == 0
        np.testing.assert_array_equal(rb.buf, data[-100:])

    def test_write_empty(self):
        '''Test writing empty array.'''
        rb = RingBuffer(100)
        rb.write(np.array([], dtype=np.float32))
        assert rb.write_idx == 0

    def test_read_latest(self):
        '''Test reading latest samples.'''
        rb = RingBuffer(100)
        data = np.arange(50, dtype=np.float32)
        rb.write(data)

        # Read last 20 samples
        result = rb.read_latest(20)
        np.testing.assert_array_equal(result, data[-20:])

    def test_read_latest_with_wrap(self):
        '''Test reading after wrap around.'''
        rb = RingBuffer(100)
        # Fill buffer and wrap
        data1 = np.arange(80, dtype=np.float32)
        rb.write(data1)
        data2 = np.arange(80, 110, dtype=np.float32)
        rb.write(data2)

        # Read last 50 samples - should cross the wrap boundary
        result = rb.read_latest(50)
        expected = np.arange(60, 110, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_read_latest_preallocated(self):
        '''Test reading into pre-allocated array.'''
        rb = RingBuffer(100)
        data = np.arange(50, dtype=np.float32)
        rb.write(data)

        out = np.empty(20, dtype=np.float32)
        result = rb.read_latest(20, out=out)
        np.testing.assert_array_equal(result, data[-20:])
        # Verify result shares memory with out buffer
        np.testing.assert_array_equal(out, data[-20:])

    def test_read_latest_more_than_available(self):
        '''Test reading more samples than buffer size.'''
        rb = RingBuffer(100)
        data = np.arange(50, dtype=np.float32)
        rb.write(data)

        # Request more than buffer size
        result = rb.read_latest(200)
        assert len(result) == 100  # Capped at buffer size

    def test_resize_expand(self):
        '''Test expanding buffer size.'''
        rb = RingBuffer(100)
        data = np.arange(100, dtype=np.float32)
        rb.write(data)

        rb.resize(200)
        assert rb.size == 200
        assert rb.write_idx == 0

        # Old data should be at the end
        result = rb.read_latest(100)
        np.testing.assert_array_equal(result, data)

    def test_resize_shrink(self):
        '''Test shrinking buffer size.'''
        rb = RingBuffer(100)
        data = np.arange(100, dtype=np.float32)
        rb.write(data)

        rb.resize(50)
        assert rb.size == 50
        assert rb.write_idx == 0

        # Only most recent 50 samples should remain
        result = rb.read_latest(50)
        np.testing.assert_array_equal(result, data[-50:])

    def test_resize_preserves_order_after_wrap(self):
        '''Test that resize preserves data order after wrap around.'''
        rb = RingBuffer(100)
        # Write data that wraps
        data1 = np.arange(80, dtype=np.float32)
        rb.write(data1)
        data2 = np.arange(80, 130, dtype=np.float32)
        rb.write(data2)

        # Resize
        rb.resize(150)

        # Check that latest data is preserved in order
        result = rb.read_latest(100)
        expected = np.arange(30, 130, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_thread_safety(self):
        '''Test concurrent read/write operations.'''
        import threading

        rb = RingBuffer(10000)
        errors = []

        def writer():
            for i in range(100):
                data = np.random.randn(100).astype(np.float32)
                rb.write(data)

        def reader():
            for i in range(100):
                try:
                    result = rb.read_latest(100)
                    assert len(result) == 100
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestSTFT:
    '''Tests for STFT computation (requires Qt app).'''

    @pytest.fixture
    def app(self):
        '''Create Qt application for testing.'''
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    @pytest.fixture
    def viz(self, app):
        '''Create RealtimeAudioViz instance.'''
        from rt_wav_sgram import RealtimeAudioViz
        viz = RealtimeAudioViz(sr=16000, seconds=2.0, blocksize=256)
        yield viz
        viz.close()

    def test_stft_output_shape(self, viz):
        '''Test STFT output has correct shape.'''
        # Generate test signal
        n_samples = viz.n_samples
        x = np.random.randn(n_samples).astype(np.float32)

        result = viz.compute_stft_mag(x)

        assert result is not None
        assert result.shape[0] == viz.freq_bins  # frequency bins
        assert result.shape[1] > 0  # time frames

    def test_stft_sine_wave(self, viz):
        '''Test STFT correctly identifies sine wave frequency.'''
        # Generate 1000 Hz sine wave
        freq = 1000
        t = np.linspace(0, viz.seconds, viz.n_samples, endpoint=False)
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)

        result = viz.compute_stft_mag(x)

        # Find peak frequency bin
        freq_resolution = viz.sr / viz.n_fft
        expected_bin = int(freq / freq_resolution)

        # Average across time frames
        avg_spectrum = result.mean(axis=1)
        peak_bin = np.argmax(avg_spectrum)

        # Should be within 2 bins of expected
        assert abs(peak_bin - expected_bin) <= 2, \
            f"Peak at bin {peak_bin}, expected ~{expected_bin}"

    def test_stft_db_range(self, viz):
        '''Test STFT output is in reasonable dB range.'''
        x = np.random.randn(viz.n_samples).astype(np.float32) * 0.1
        result = viz.compute_stft_mag(x)

        # Output should be in dB (negative values for quiet signals)
        assert result.max() < 20  # Not unreasonably high
        assert result.min() > -200  # Not unreasonably low

    def test_stft_short_input(self, viz):
        '''Test STFT handles input shorter than FFT size.'''
        x = np.random.randn(viz.n_fft // 2).astype(np.float32)
        result = viz.compute_stft_mag(x)

        # Should still produce valid output (padded)
        assert result is not None
        assert result.shape[0] == viz.freq_bins


class TestControlSidebar:
    '''Tests for the ControlSidebar widget.'''

    @pytest.fixture
    def app(self):
        '''Create Qt application for testing.'''
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    @pytest.fixture
    def sidebar(self, app):
        '''Create ControlSidebar instance.'''
        from rt_wav_sgram import ControlSidebar
        sidebar = ControlSidebar()
        yield sidebar
        sidebar.close()

    def test_default_values(self, sidebar):
        '''Test sidebar has correct default values.'''
        assert sidebar.db_min_spin.value() == -50.0
        assert sidebar.db_max_spin.value() == 0.0
        assert sidebar.max_freq_spin.value() == 6000
        assert sidebar.fps_spin.value() == 60
        assert sidebar.wave_gain_spin.value() == 4.0
        assert sidebar.duration_spin.value() == 4.0

    def test_db_range_signal(self, sidebar, qtbot):
        '''Test dB range signal emission.'''
        from pytestqt.qtbot import QtBot

        with qtbot.waitSignal(sidebar.db_range_changed, timeout=1000) as blocker:
            sidebar.db_min_spin.setValue(-60.0)

        assert blocker.args == [-60.0, 0.0]

    def test_colormap_signal(self, sidebar, qtbot):
        '''Test colormap signal emission.'''
        with qtbot.waitSignal(sidebar.colormap_changed, timeout=1000) as blocker:
            sidebar.colormap_combo.setCurrentText('viridis')

        assert blocker.args == ['viridis']

    def test_stream_settings_signal(self, sidebar, qtbot):
        '''Test stream settings signal emission.'''
        with qtbot.waitSignal(sidebar.stream_settings_changed, timeout=1000) as blocker:
            sidebar.apply_btn.click()

        settings = blocker.args[0]
        assert 'sr' in settings
        assert 'seconds' in settings
        assert 'blocksize' in settings
        assert 'n_fft' in settings
        assert 'hop' in settings

    def test_button_states(self, sidebar):
        '''Test start/stop button states.'''
        # Initially: start enabled, stop disabled
        assert sidebar.start_btn.isEnabled()
        assert not sidebar.stop_btn.isEnabled()

        # After starting: start disabled, stop enabled
        sidebar.set_stream_running(True)
        assert not sidebar.start_btn.isEnabled()
        assert sidebar.stop_btn.isEnabled()

        # After stopping: back to initial state
        sidebar.set_stream_running(False)
        assert sidebar.start_btn.isEnabled()
        assert not sidebar.stop_btn.isEnabled()


class TestRealtimeAudioViz:
    '''Tests for the main RealtimeAudioViz window.'''

    @pytest.fixture
    def app(self):
        '''Create Qt application for testing.'''
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    @pytest.fixture
    def viz(self, app):
        '''Create RealtimeAudioViz instance.'''
        from rt_wav_sgram import RealtimeAudioViz
        viz = RealtimeAudioViz(sr=16000, seconds=2.0, blocksize=256)
        yield viz
        viz.stop_stream()
        viz.close()

    def test_init_parameters(self, viz):
        '''Test initial parameters are set correctly.'''
        assert viz.sr == 16000
        assert viz.seconds == 2.0
        assert viz.blocksize == 256
        assert viz.n_samples == 32000  # 16000 * 2.0

    def test_parameter_handlers(self, viz):
        '''Test real-time parameter handlers.'''
        # dB range
        viz._on_db_range_changed(-70.0, -10.0)
        assert viz.db_min == -70.0
        assert viz.db_max == -10.0

        # Wave gain
        viz._on_wave_gain_changed(8.0)
        assert viz.wave_gain == 8.0

        # Max frequency
        viz._on_max_freq_changed(4000)
        assert viz.display_max_freq == 4000

    def test_max_freq_bounded_by_nyquist(self, viz):
        '''Test max frequency is bounded by Nyquist.'''
        # Try to set max freq higher than Nyquist (8000 Hz at 16kHz SR)
        viz._on_max_freq_changed(10000)
        assert viz.display_max_freq == viz.max_freq  # Should be capped at Nyquist

    def test_stream_settings_change(self, viz):
        '''Test changing stream settings.'''
        new_settings = {
            'sr': 16000,
            'seconds': 3.0,
            'blocksize': 512,
            'n_fft': 512,
            'hop': 128,
        }
        viz._on_stream_settings_changed(new_settings)

        assert viz.seconds == 3.0
        assert viz.blocksize == 512
        assert viz.n_fft == 512
        assert viz.hop == 128
        assert viz.n_samples == 48000  # 16000 * 3.0

    def test_buffer_resize_on_duration_change(self, viz):
        '''Test that buffer is resized when duration changes.'''
        # Write some data
        test_data = np.arange(1000, dtype=np.float32)
        viz.ring.write(test_data)

        # Change duration
        new_settings = {
            'sr': 16000,
            'seconds': 3.0,
            'blocksize': 256,
            'n_fft': 256,
            'hop': 64,
        }
        viz._on_stream_settings_changed(new_settings)

        # Buffer should be resized
        assert viz.ring.size == 48000

        # Data should be preserved
        result = viz.ring.read_latest(1000)
        np.testing.assert_array_equal(result, test_data)

    def test_colormap_grayscale(self, viz):
        '''Test grayscale colormap application.'''
        viz._on_colormap_changed('grayscale')
        # Should not raise an error

    def test_colormap_builtin(self, viz):
        '''Test built-in colormap application.'''
        viz._on_colormap_changed('viridis')
        # Should not raise an error

    def test_update_plots_no_crash(self, viz):
        '''Test update_plots doesn't crash with empty buffer.'''
        viz.update_plots()
        # Should complete without error

    def test_update_plots_with_data(self, viz):
        '''Test update_plots with data in buffer.'''
        # Write test data
        test_data = np.random.randn(viz.n_samples).astype(np.float32) * 0.5
        viz.ring.write(test_data)

        # Update should complete without error
        viz.update_plots()


class TestPerformance:
    '''Tests for performance optimizations.'''

    def test_scipy_fft_available(self):
        '''Test that scipy FFT is available for faster computation.'''
        from rt_wav_sgram import USE_SCIPY_FFT
        assert USE_SCIPY_FFT is True, "scipy.fft should be available"

    @pytest.fixture
    def app(self):
        '''Create Qt application for testing.'''
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    @pytest.fixture
    def viz(self, app):
        '''Create RealtimeAudioViz instance.'''
        from rt_wav_sgram import RealtimeAudioViz
        viz = RealtimeAudioViz(sr=16000, seconds=2.0, blocksize=256)
        yield viz
        viz.stop_stream()
        viz.close()

    def test_preallocated_buffers_exist(self, viz):
        '''Test that pre-allocated buffers are created.'''
        assert hasattr(viz, 'fft_buffer')
        assert hasattr(viz, 'spec_display_buffer')
        assert hasattr(viz, 'wave_display_buffer')
        assert viz.fft_buffer.dtype == np.complex64
        assert viz.spec_display_buffer.dtype == np.float32

    def test_fps_update_counter(self, viz):
        '''Test FPS update counter initialization.'''
        assert hasattr(viz, '_fps_update_counter')
        assert viz._fps_update_counter == 0


class TestAudioCheck:
    '''Tests for audio availability check.'''

    def test_check_audio_available_returns_tuple(self):
        '''Test that check_audio_available returns a tuple.'''
        from rt_wav_sgram import check_audio_available
        result = check_audio_available()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_check_audio_available_message(self):
        '''Test that check_audio_available provides a message.'''
        from rt_wav_sgram import check_audio_available
        available, message = check_audio_available()
        # Message should be non-empty regardless of availability
        assert len(message) > 0


class TestErrorHandling:
    '''Tests for error handling in RealtimeAudioViz.'''

    @pytest.fixture
    def app(self):
        '''Create Qt application for testing.'''
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    @pytest.fixture
    def viz(self, app):
        '''Create RealtimeAudioViz instance.'''
        from rt_wav_sgram import RealtimeAudioViz
        viz = RealtimeAudioViz(sr=16000, seconds=2.0, blocksize=256)
        yield viz
        viz.stop_stream()
        viz.close()

    def test_show_error_method_exists(self, viz):
        '''Test that _show_error method exists.'''
        assert hasattr(viz, '_show_error')
        assert callable(viz._show_error)

    def test_show_warning_method_exists(self, viz):
        '''Test that _show_warning method exists.'''
        assert hasattr(viz, '_show_warning')
        assert callable(viz._show_warning)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
