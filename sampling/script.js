window.addEventListener('load', () => {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    const downloadBtn = document.getElementById('downloadBtn');
    const clearBtn = document.getElementById('clearBtn');

    // --- グリッド設定 ---
    const GRID_SIZE = 8; // 8x8
    let cellSize = 0;
    // -------------------

    // Canvasのサイズを「正方形」に設定
    const controlsHeight = document.getElementById('controls').offsetHeight;
    const availableHeight = window.innerHeight - controlsHeight;
    // 利用可能な幅と高さのうち、小さい方に合わせて正方形を作成
    const size = Math.min(window.innerWidth - 4, availableHeight - 4); // 枠線やマージンを考慮
    
    canvas.width = size;
    canvas.height = size;
    
    // セルサイズを計算
    cellSize = canvas.width / GRID_SIZE;

    // グリッド線を描画する関数
    function drawGrid() {
        ctx.strokeStyle = '#e0e0e0'; // 薄いグレー
        ctx.lineWidth = 0.5;
        for (let i = 1; i < GRID_SIZE; i++) {
            // 縦線
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, canvas.height);
            ctx.stroke();
            // 横線
            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(canvas.width, i * cellSize);
            ctx.stroke();
        }
        // 描画用の設定に戻す
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
    }

    // 初期グリッド描画
    drawGrid();

    // 描画用の設定
    ctx.lineWidth = 2; // 線の太さ
    ctx.strokeStyle = '#000000'; // 線の色
    ctx.lineCap = 'round'; // 線の端を丸く
    ctx.lineJoin = 'round'; // 線の角を丸く

    let isDrawing = false;
    let strokeCount = 0; // ストローク番号
    let drawingData = []; // 全ての座標データを格納する配列

    // ----------------------------------------
    // イベントリスナー
    // ----------------------------------------
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    downloadBtn.addEventListener('click', downloadCSV);
    clearBtn.addEventListener('click', clearCanvas);

    // ----------------------------------------
    // 描画とデータ記録の関数
    // ----------------------------------------

    function startDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = true;
        strokeCount++;
        
        const pos = getCanvasPosition(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        recordData(e, 'down');
    }

    function draw(e) {
        if (!isDrawing || e.pointerType !== 'pen') return;
        const pos = getCanvasPosition(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        recordData(e, 'move');
    }

    function stopDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = false;
    }

    function getCanvasPosition(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    // データを配列に格納 (★修正点)
    function recordData(e, eventType) {
        const pos = getCanvasPosition(e);

        // 座標からセルIDを計算
        const cellX = Math.floor(pos.x / cellSize);
        const cellY = Math.floor(pos.y / cellSize);
        
        // 範囲外（0未満や8以上）にならないように補正
        const validCellX = Math.max(0, Math.min(cellX, GRID_SIZE - 1));
        const validCellY = Math.max(0, Math.min(cellY, GRID_SIZE - 1));
        
        // セルID (0〜63)
        // (例: 0行目の0列目 = 0, 1行目の0列目 = 8)
        const cellId = (validCellY * GRID_SIZE) + validCellX;

        drawingData.push({
            timestamp: e.timeStamp,
            event_type: eventType,
            stroke_id: strokeCount,
            x: pos.x,
            y: pos.y,
            pressure: e.pressure,
            tilt_x: e.tiltX,
            tilt_y: e.tiltY,
            // --- ★追加データ ---
            cell_x: validCellX, // どの列か (0-7)
            cell_y: validCellY, // どの行か (0-7)
            cell_id: cellId     // どのセルか (0-63)
            // -------------------
        });
    }

    // ----------------------------------------
    // CSV処理とクリアの関数
    // ----------------------------------------

    function downloadCSV() {
        if (drawingData.length === 0) {
            alert("データがありません。");
            return;
        }

        // CSVのヘッダー行 (★修正点)
        const headers = "timestamp,event_type,stroke_id,x,y,pressure,tilt_x,tilt_y,cell_x,cell_y,cell_id";
        
        // CSVのデータ行を作成 (★修正点)
        const csvRows = drawingData.map(d => {
            return [
                d.timestamp,
                d.event_type,
                d.stroke_id,
                d.x.toFixed(2),
                d.y.toFixed(2),
                d.pressure.toFixed(4),
                d.tilt_x,
                d.tilt_y,
                // --- ★追加データ ---
                d.cell_x,
                d.cell_y,
                d.cell_id
                // -------------------
            ].join(',');
        });

        const csvContent = headers + "\n" + csvRows.join("\n");
        const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
        const blob = new Blob([bom, csvContent], { type: 'text/csv;charset=utf-8;' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'unpitsu_data_grid.csv'; // (ファイル名変更)
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function clearCanvas() {
        if (confirm("データを消去しますか？")) {
            drawingData = [];
            strokeCount = 0;
            // Canvasをクリア
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // グリッド線を再描画
            drawGrid();
        }
    }
});