def dot(a, b):
    if (len(a) != len(b)): raise Exception("The lenghts of the two arrays don't match")
    return sum([x * y for x, y in zip(a, b)])


def identityMatrix(size):
    return Matrix([[1 if x == y else 0 for y in range(size)] for x in range(size)])

def zeroMat(rows, cols):
    return Matrix([[0 for x in range(cols)] for y in range(rows)])




class stringF(object):
    def __init__(self, x):
        self.x = x

    def __mul__(self, other):
        if isinstance(other, stringF):
            return stringF("(" + self.x + " * " +  other.x + ")")

    def __rmul__(self, other):
        if isinstance(other, stringF):
            return stringF("(" + other.x + " * " +  self.x + ")")

    def __add__(self, other):
        if isinstance(other, stringF):
            return stringF("(" + self.x + " + " +  other.x + ")")


    def __sub__(self, other):
        if isinstance(other, stringF):
            return stringF("(" + self.x + " - " +  other.x + ")")

    def __str__(self):
        return self.x


class Matrix(object):


    def __init__(self, arr):
        self.arr = arr
        self.rowCount = len(arr)
        self.colCount = len(arr[0])

    def __eq__(self, other):

        return isinstance(other, Matrix) and self.rowCount == other.rowCount and self.colCount == other.colCount and self.arr == other.arr

    def fold(self, base, f):
        """Peform some kind of accumulation on the elements of the matrix."""
        current = base
        for row in self.arr:
            for element in row:
                current = f(current, element)
        return current

    def transpose(self):
        tMat = zeroMat(self.colCount, self.rowCount)
        for rowIndex in range(self.rowCount):
            for colIndex in range(self.colCount):
                tMat[colIndex, rowIndex] = self[rowIndex, colIndex]
        return tMat

    def reshape(self, rows, cols):
        """Change the shape of the matrix"""
        if (rows * cols != self.rowCount * self.colCount): raise Exception("Reshaped matrix must have the same element number!")
        total = []    
        for row in self.arr:
            total += row
        nmat = []
        for ri in range(rows):
            row = []
            for ci in range(cols):
                row.append(total[ri * cols + ci])
            nmat.append(row)
        return Matrix(nmat)
        """"
        mat = zeroMat(rows, cols)
        for row in range(self.rowCount):
            for col in range(self.colCount):
                total = row * self.colCount + col
                nRow = math.floor(total/rows)
                nCol = total - nRow
                print(nRow, nCol)
                mat[nRow, nCol] = self[row, col]
        """

    def getCopy(self):
        return Matrix([[x for x in row] for row in self.arr])

    ##def isSameAs(self, other):
    def setDiagonalOfThis(self, elements):
        """Set the diagonal of the matrix to the values"""
        if (self.rowCount != self.colCount): raise Exception("Can't set diagonal on non square matrix!")
        if (len(elements) != self.colCount): raise Exception("Element count must be equal to diagonal count!")
        for index, value in enumerate(elements):
            self[index,index] = value


    def getElement(self, rowIndex, colIndex):
        """Returns an element in the matrix."""
        return self.arr[rowIndex][colIndex]

    def __setitem__(self, rowColIndex, value):
        row, col = rowColIndex 
        self.arr[row][col] = value

    def __getitem__(self, rowColIndex):
        row, col = rowColIndex 
        return self.arr[row][col]

    def getRow(self, rowIndex):
        """Returns the requested row of the matrix."""
        return self.arr[rowIndex]

    def isSquare(self):
        """Is the matrix square?"""
        return self.rowCount == self.colCount

    def determinant(self):
        """Calculate the determinant of this matrix incredibly slowly (n!)."""
        if (not self.isSquare()): raise Exception("Can't calculate determinant of non square matrix!")
        return self._calculateDeterminatneInternal(self)
    
    def _calculateDeterminatneInternal(self, mat):
        """An incredibly slow method for calculating the determinant."""
        if  (mat.rowCount == 0): raise Exception("Matrix was empty!")
        elif (mat.rowCount == 1): return mat[9, 0]
        elif (mat.rowCount == 2):
            return mat[0,0] * mat[1,1] - mat[0, 1] * mat[1,0]

        baseMat = zeroMat(mat.colCount-1, mat.rowCount-1)
        total = 0
        modFlip = 1
        for i in range(mat.colCount):
            for j in range(mat.colCount):
                if (i != j):
                    for k in range(1, mat.colCount):
                        baseJ = j if j < i else j-1
                        baseMat[k-1, baseJ] = mat[k, j]
            total += mat[0, i] * self._calculateDeterminatneInternal(baseMat) * modFlip
            modFlip *= -1
        return total


    def gausianElimination(self):
        """
        Perform gaussian elimination, to echelon form, and return a new matrix.
        """
        return self._gausianElimination(self.getCopy())
    
    def toEchelonForm(self):
        """Return a new matrix in echelon form."""
        return self.gausianElimination()

    def toReducedRowEchelonForm(self):
        """Return a new matrix in reduced row echelon form."""
        # Convert first to echelon form, then reduce from there.
        mat = self.toEchelonForm()

        # Zeros all numbers in the matrix that are not on the diagonal.
        # assuming the matrix is square. It still works if the matrix is not square,
        # it just won't make those cells outside the square matrix zero.
        for subRowIndex in range(mat.rowCount-1, 0, -1):
            # This is the row that we'll subtract from the rows above it by some multiplier.
            subRow = mat.arr[subRowIndex]
            nonZeroIndex = self._firstNonZeroIndex(subRow)
            if (nonZeroIndex == None): continue#raise Exception("Can't reduce the matrix because a row is all zeros!")
            subRowElementNonZero = subRow[nonZeroIndex]
            for rowIndex in range(0, subRowIndex):
                row = mat.arr[rowIndex]
                # The multiplier which will zero the element at the index "nonZeroIndex" of
                # each row above the subRowIndex
                scale = row[nonZeroIndex]/subRowElementNonZero
                for columnIndex, subRowElement in enumerate(subRow):
                    row[columnIndex] -= subRowElement * scale

        # Converts all numbers on the diagonal to 1.
        for row in mat.arr:
            nonZeroIndex = self._firstNonZeroIndex(row)
            if nonZeroIndex != None:
                firstNonZero = row[nonZeroIndex]
                for index, element in enumerate(row):
                    row[index] = element/firstNonZero
        return mat 
            
    def appendOtherMatrixToRight(self, otherMatrix):
        """Append a matrix to the right of this matrix."""
        if (self.rowCount != otherMatrix.rowCount): raise Exception("The matrices must have the same row count!")
        return Matrix([self.arr[rowIndex] + otherMatrix.arr[rowIndex] for rowIndex in range(self.rowCount)])

    def getSubMatrix(self, startRow, endRow, startCol, endCol):
        """Produce a sub matrix of this matrix, with the appropriate size."""
        subMat = [[self[row, col] for col in range(startCol, endCol)] for row in range(startRow, endRow)]
        return Matrix(subMat)

    def invert(self):
        """Invert the matrix."""
        if (self.rowCount != self.colCount): raise Exception("The matrix must be square to invert!")

        ident = identityMatrix(self.rowCount)
        augumentedMatrix = self.appendOtherMatrixToRight(ident)
        solvedMatrix = augumentedMatrix.toReducedRowEchelonForm()
        return solvedMatrix.getSubMatrix(0, self.rowCount, self.colCount, self.colCount + self.colCount)


    def swapRowsInPlace(self, row1Index, row2Index):
        """Swaps two rows in place."""
        row1tmp = self.arr[row1Index]
        self.arr[row1Index] = self.arr[row2Index]
        self.arr[row2Index] = row1tmp

    def _scaleList(self, xs, scaler):
        for index, value in enumerate(xs):
            xs[index] *= value

    def multiplyRowInPlace(self, rowIndex, scaler):
        """Multiply a row in place"""
        self._scaleList(self.arr[rowIndex], scaler)        

    def addToRowInPlace(self, rowIndex, toAdd):
        #self.arr[rowIndex] = [a + b for a, b in zip(self.arr[rowIndex, toAdd])]
        for colIndex, value in enumerate(toAdd):
            self.arr[rowIndex][colIndex] += value


    def _firstNonZeroIndex(self, xs):
        """ Returns the index of the first non zero element, or None. """
        for index, value in enumerate(xs):
            if (value != 0): return index
        return None


    def _findRowWithNonZero(self, startRow):
        """
        Returns the first row, from start row, which contains a
        non-zero element, or None if all rows are zero.
        """
        for index, row in enumerate(self.arr[startRow:]):
            rowIndex = startRow + index
            firstNonZero = self._firstNonZeroIndex(row)
            if (firstNonZero is not None): return rowIndex
        return None

    def __add__(self, otherMatrix):
        if (not isinstance(otherMatrix, Matrix)): 
            raise Exception("Can only add two matrices!")
        if (not (otherMatrix.rowCount == self.rowCount and otherMatrix.colCount == self.colCount)): 
            raise Exception("Matrices must be same size!")
        at = [[self[row, col] + otherMatrix[row, col] for col in range(self.colCount)] for row in range(self.rowCount)]
        return Matrix(at)

    def __sub__(self, otherMatrix):
        ##if (not isinstance(otherMatrix, Matrix)): 
        #    raise Exception("Can only subtract two matrices!")
        if (not (otherMatrix.rowCount == self.rowCount and otherMatrix.colCount == self.colCount)): 
            raise Exception("Matrices must be same size!")
        at = [[self[row, col] - otherMatrix[row, col] for col in range(self.colCount)] for row in range(self.rowCount)]
        return Matrix(at)

    def mapInPlace(self, f):
        """Update the elements of this matrix with a function f(x)."""
        for row in self.arr:
            for index, element in enumerate(row):
                row[index] = f(element)
        return self

    def mapToNewMatrix(self, f):
        """Returns a new matrix, running f(x) on each element of this one."""
        return Matrix([[f(x) for x in row] for row in self.arr])
    def getSize(self):
        return (self.rowCount, self.colCount)
    def elementMul(self, other):
        if (self.rowCount != other.rowCount or self.colCount != other.colCount): raise Exception("Can't do elementwise multiplication on vectors with different dimensions!")
        out = other.getCopy()
        for i in range(self.rowCount):
            for j in range(self.colCount):
                out[i, j] *= self[i, j]
        return out


    def _gausianElimination(self, mat):
        """ 
        Perform gaussian elimination on the passed in matrix.
        Stop once echelon form is reached
        """
        #lastZeroRowIndex = mat.rowCount-1

        #zeroRowIndexes = [False for x in range(mat.rowCount)]

        i = 0
        while (i < mat.rowCount-1):
            ithRow = mat.arr[i]
            iFirstNonZeroIndex = mat._firstNonZeroIndex(ithRow)
            if (iFirstNonZeroIndex is None):
                #if (i == mat.rowCount -1)
                #if (lastZeroRowIndex <= 0): return mat
                rowIndex = mat._findRowWithNonZero(i)
                if rowIndex is None: return mat
                mat.swapRowsInPlace(i, rowIndex)
                
                #lastZeroRowIndex -= 1

                continue

            for j in range(i + 1, mat.rowCount):
                jthRow = mat.arr[j]
                jFirstNonZeroIndex = mat._firstNonZeroIndex(jthRow)

                if (jFirstNonZeroIndex is None):
                    if (j == mat.rowCount-1): return mat
                    rowIndex = mat._findRowWithNonZero(j)
                    if rowIndex is None: return mat
                    mat.swapRowsInPlace(j, j + 1)
                    i-=1
                    break

                # Have to swap rows here, because the first non zero index of the ith row
                # is greater than the jth row.
                if (iFirstNonZeroIndex > jFirstNonZeroIndex):
                    mat.swapRowsInPlace(i, j)
                    # subtract one from i so we can rexamine the flipped row
                    i -= 1
                    break
                elif (iFirstNonZeroIndex == jFirstNonZeroIndex):
                    ithVal = ithRow[iFirstNonZeroIndex]
                    jthVal = jthRow[jFirstNonZeroIndex]
                    # the scale just makes it so we can zero the first
                    # non zero column of the jth row.
                    scale = -jthVal/ithVal
                    mat.addToRowInPlace(j, [x * scale for x in ithRow])
            i+=1
        return mat
            
    def getColumn(self, columnIndex):
        """Returns the requested column of this matrix"""
        #if (columnIndex > self.rowCount or columnIndex < 0): raise Exception("Column index out of bounds!")
        return [row[columnIndex] for row in self.arr]
    
    def getSizeAsString(self):
        return "(" + str(self.rowCount) + ", " + str(self.colCount) + ")"

    def multiplyMats(self, a, b):
        """Multiply two matrices together."""
        if (a.colCount != b.rowCount): raise Exception("Matrices are mismatched! a: " + a.getSizeAsString() + ", b: " + b.getSizeAsString())
        mulArr = []
        for i in range(a.rowCount):
            inner = []
            for j in range(b.colCount):
                #inner.append(Dot(a.getRow(i), b.getColumn(j)))
                total = 0
                for k in range(a.colCount):
                    total += a[i, k] * b[k, j]
                inner.append(total)
            mulArr.append(inner)
        return Matrix(mulArr)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            newArray = [[r * other for r in row] for row in self.arr]
            return Matrix(newArray)
        elif isinstance(other, Matrix):
            return self.multiplyMats(self, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            newArray = [[r * other for r in row] for row in self.arr]
            return Matrix(newArray)
        if (isinstance(other, Matrix)):
            return self.multiplyMats(other, self)
        
    maxStringCount = 10
    printSF = 5
    printRjust = 11

    def _formatPrintElement(self, element, sf, rightAdjust):
        elementVal = element
        if isinstance(element, int):
            elementVal = str(element)
        elif not isinstance(element, str):
            elementVal = str.format("{0:." + str(sf) + "}", element)
            #elementVal = str(round(element, sf))
        return elementVal.rjust(rightAdjust, " ")

    def _getListAsString(self, xs):

        if self.colCount > self.maxStringCount:
            halfSize = int(self.maxStringCount/2)
            start = ", ".join([self._formatPrintElement(x, self.printSF, self.printRjust) for x in xs[:halfSize]])
            end = ", ".join([self._formatPrintElement(x, self.printSF, self.printRjust) for x in xs[-halfSize:]])
            return start + " . . . " + end
        else: 
            return ", ".join([self._formatPrintElement(x, self.printSF, self.printRjust) for x in xs])
    
    def __str__(self):

        if (self.rowCount > self.maxStringCount):
            halfSize = int(self.maxStringCount/2)
            start = "\n".join([self._getListAsString(row) for row in self.arr[:halfSize]])
            end = "\n".join([self._getListAsString(row) for row in self.arr[-halfSize:]])
            midRow = "\n".join([self._getListAsString(["...  " for x in range(self.colCount)]) for t in range(3)])

            return "\n".join([start, midRow, end])
        else:
            return "\n".join([self._getListAsString(row) for row in self.arr])

