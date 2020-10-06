import numpy as np

def returnPointsOnEdgeOfCircleSquareUnion(x0,y0,r,S):
    from numpy import sqrt, asarray, unique
    vertices = []
    if (r**2 > y0**2): # Intersect line y = 0
        # Check points are acceptable
        a = x0+sqrt(r**2-y0**2)
        b = x0-sqrt(r**2-y0**2)
        if a < S:
            vertices.append([a,0])
        else:
            vertices.append([S,0])
        
        if b > 0:
            vertices.append([b,0])
        else:
            vertices.append([0,0])
    
    if (r**2 > x0**2): # Intersect line x = 0
        # Check points are acceptable
        a = y0+sqrt(r**2-x0**2)
        b = y0-sqrt(r**2-x0**2)
        if a < S:
            vertices.append([0,a])
        else:
            vertices.append([0,S])
        
        if b > 0:
            vertices.append([0,b])
        else:
            vertices.append([0,0])
            
    if (r**2 > (S-y0)**2): # Intersect line y = S
        # Check points are acceptable
        a = x0+sqrt(r**2-(S-y0)**2)
        b = x0-sqrt(r**2-(S-y0)**2)
        if a < S:
            vertices.append([a,S])
        else:
            vertices.append([S,S])
        
        if b > 0:
            vertices.append([b,S])
        else:
            vertices.append([0,S])

    if (r**2 > (S-x0)**2): # Intersect line x = S
        # Check points are acceptable
        a = y0+sqrt(r**2-(S-x0)**2)
        b = y0-sqrt(r**2-(S-x0)**2)
        if a < S:
            vertices.append([S,a])
        else:
            vertices.append([S,S])
        
        if b > 0:
            vertices.append([S,b])
        else:
            vertices.append([S,0])
            
    if len(vertices) > 0:
        vertices = unique(asarray(vertices),axis=0)
    return vertices

def returnAreaOfCircleWithinDomain(x,y,r,S):
    import numpy as np
    vertices = returnPointsOnEdgeOfCircleSquareUnion(x,y,r,S)
    numVert = np.shape(vertices)[0]
    # The set of vertices traces the boundary of the union of square and circle
    # If two vertices share a boundary of the square, we calculate area of triangle within
    sectorAngle = 2*np.pi
    area = 0
    a = [x,y]
    for vi in range(numVert):
        for vj in range(vi+1,numVert):
            if any(vertices[vi] == vertices[vj]):
                # Test if the matching vertex is 0 or S
                match = vertices[vi,vertices[vi]==vertices[vj]]
                if float(match) in [0,S]:
                    b = vertices[vi]
                    c = vertices[vj]
                    # The vertices share an edge. Calculate triangle area
                    area = area + np.abs( 0.5*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])))
                    
                    # Calculate interior angle (http://www.ambrsoft.com/TrigoCalc/Triangles/3Points.htm)
                    vec1 = b - a
                    vec2 = c - a
                    dotProduct = vec1[0]*vec2[0] + vec1[1]*vec2[1]
                    mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
                    mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
                    
                    intAngle = np.arccos(dotProduct/(mag1*mag2))
                    sectorAngle = sectorAngle - intAngle
        
    # Now calculate area of remaining sectors based on angle
    assert(sectorAngle >= -1e-4)
    remainingArea = 0.5 * sectorAngle * r**2
    area = area + remainingArea
    return area


def pairCorrelationFunction_2D_edgeEffects(x, y, S, rMax, dr):
    """Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane. This is an attempt to compensate for edge effects!
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        dr              increment for increasing radius of annulus
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """
    import numpy as np
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram
    from sklearn.neighbors import KDTree
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    
    # Store distances of each point from each edge of the square
    
    # For each r, g(r) is average density of points in annulus of width r, 
    # but some annuli will be incomplete as close to the boundary
    
    # Can therefore calculate g(r) for some r by finding total area of annuli to include
    # and total number of points separated by that distance (x2 as double counts)
    x = np.asarray(x)
    y = np.asarray(y)
    pointsToTest = np.vstack((x,y)).transpose()
    Y = cdist(pointsToTest, pointsToTest, 'euclidean')
    
    densityOfPointsToPlace = len(x) / S**2
    
    counts = []    
    radii = np.arange(0,rMax+dr,dr)
    for r in range(len(radii)-1):
        inner = radii[r]
        outer = radii[r+1]
        counts.append( sum(sum((Y > inner) & (Y <= outer))) )

        
    counts = np.asarray(counts)
    areas = np.zeros((len(x),len(counts)))
    for pointInd in range(len(x)):
        for r in range(len(radii)-1):
            inner = radii[r]
            outer = radii[r+1]
            
            areaOuter = returnAreaOfCircleWithinDomain(x[pointInd],y[pointInd],outer,S)
            areaInner = returnAreaOfCircleWithinDomain(x[pointInd],y[pointInd],inner,S)
            areas[pointInd,r] = areaOuter - areaInner
            
    expPoints = densityOfPointsToPlace * areas
    avgExpPoints = np.mean(expPoints,axis=0)
    g_average = (counts/len(x))/avgExpPoints
    radii = radii[0:-1]
    
    return (g_average, radii)

def pairCorrelationFunction_2D_edgeEffectsParallel(x, y, S, rMax, dr):
    """Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane. This is an attempt to compensate for edge effects!
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        dr              increment for increasing radius of annulus
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """
    import numpy as np
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram
    from sklearn.neighbors import KDTree
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    import multiprocessing
    from joblib import Parallel, delayed

    num_cores = multiprocessing.cpu_count()

    
    # Store distances of each point from each edge of the square
    
    # For each r, g(r) is average density of points in annulus of width r, 
    # but some annuli will be incomplete as close to the boundary
    
    # Can therefore calculate g(r) for some r by finding total area of annuli to include
    # and total number of points separated by that distance (x2 as double counts)
    x = np.asarray(x)
    y = np.asarray(y)
    pointsToTest = np.vstack((x,y)).transpose()
    Y = cdist(pointsToTest, pointsToTest, 'euclidean')
    
    densityOfPointsToPlace = len(x) / S**2
    
    counts = []    
    radii = np.arange(0,rMax+dr,dr)
    for r in range(len(radii)-1):
        inner = radii[r]
        outer = radii[r+1]
        counts.append( sum(sum((Y > inner) & (Y <= outer))) )

        
    counts = np.asarray(counts)
    areas = np.zeros((len(x),len(counts)))
    
    output = Parallel(n_jobs=num_cores)(delayed(parallelLoopForPCF)(radii=radii,x=x,y=y,S=S, pointInd=pointInd,r=r) for pointInd in range(len(x)) for r in range(len(radii)-1))           
    areas = np.reshape(output,(len(x),len(radii)-1))
    
    expPoints = densityOfPointsToPlace * areas
    avgExpPoints = np.mean(expPoints,axis=0)
    g_average = (counts/len(x))/avgExpPoints
    radii = radii[0:-1]
    
    return (g_average, radii)

def parallelLoopForPCF(radii,x,y,S,pointInd,r):
    inner = radii[r]
    outer = radii[r+1]
    
    areaOuter = returnAreaOfCircleWithinDomain(x[pointInd],y[pointInd],outer,S)
    areaInner = returnAreaOfCircleWithinDomain(x[pointInd],y[pointInd],inner,S)
    return (areaOuter - areaInner)


#%%
def Jfunction(pointsIn,S,bdyDist):
    import numpy as np
    from scipy.spatial.distance import cdist

    pointsToTest = pointsIn
    # First, find the SCD
    # Run same number of iterations as there are points
    randomPoints = np.random.uniform(0,S,size=(len(pointsToTest),2))
    allDists = cdist(pointsIn,randomPoints)
    maskedDists = np.ma.masked_equal(allDists,0) # Mask out 0 values
    SCD = np.amin(maskedDists,0)
       
    # Nearest neighbour
    allDistsNN = cdist(pointsIn,pointsToTest)
    maskedDistsNN = np.ma.masked_equal(allDistsNN,0) # Mask out 0 values
    NN = np.amin(maskedDistsNN,0)
   
    # Now plot the J function
    J = []
    X1 = np.sort(NN)
    nNN = np.array(range(len(NN)))/float(len(NN))
    X2 = np.sort(SCD)
    nSCD = np.array(range(len(SCD)))/float(len(SCD))
    
    maxMmRange = S
    nBins = 2000
    x = np.linspace(0,maxMmRange,nBins)
    nSCD = np.interp(x,X2,nSCD)
    nNN = np.interp(x,X1,nNN)
    
    if maxMmRange > X2[-1]:
        limInd = np.where(x > X2[-1])[0][0]
    else:
        limInd = len(x)-1
        
    J = (1 - nNN[0:limInd])/(1 - nSCD[0:limInd])
    r_J = x[0:len(J)]
    
    return r_J, J, SCD, NN

#%%
def fftIndgen(n):
    a = list(range(0, int(n/2+1)))
    b = list(range(1, int(n/2)))
    # Reverse b
    b = b[::-1]
    b = [-i for i in b]
    return a + b

def generateGRF(gridSize,domainWidthMm,targetPatternWidthMm):
    SD = targetPatternWidthMm/(2*np.sqrt(2*np.log(2))) # Via full width at half maximum - http://mathworld.wolfram.com/GaussianFunction.html
    amplitude = np.zeros((gridSize,gridSize))
    space = np.linspace(-0.5*domainWidthMm,0.5*domainWidthMm,gridSize)
    for i, kx in enumerate(space):
        for j, ky in enumerate(space):
            amplitude[i, j] = np.exp( -(kx**2 + ky**2)/(2*SD**2))    
    
    noise = np.random.normal(size = (gridSize, gridSize))
    noise_f = np.fft.fft2(noise)
    amplitude_f = np.fft.fft2(amplitude)
    amplitude_f_abs = np.abs(amplitude_f.real)
    return np.fft.ifft2(amplitude_f_abs*noise_f).real

#%%
def GenerateRandomFieldAndReturnOutputStatistics(GRF, rho, targetDensity, domainSizeMm, scaleMmToGridSize,scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr):
    import time
    targetNum = targetDensity*domainSizeMm*domainSizeMm
    gridSize = np.shape(GRF)[0]
    points = [[-1, -1]] # Add temporary point to be removed later
    numPoints = 0
    timeout = time.time() + 10 # Give it up to 10 seconds of being stuck in the loop
    while numPoints < targetNum and time.time() < timeout:
        x = np.random.rand()*(gridSize-1)
        y = np.random.rand()*(gridSize-1)
        condition = GRF[round(x),round(y)] > 0
        if condition or (np.random.rand() < rho):
            # Point is acceptable - now check that it's not within exclusionLengthscaleMm of another cell
            pointsArr = np.asarray(points)
            diffSquared = (pointsArr[:,0] - y)**2 + (pointsArr[:,1] - x)**2
            if not any(diffSquared < (exclusionLengthscaleMm*scaleMmToGridSize)**2):
                points.append([y,x])
                numPoints = numPoints + 1
    points = points[1:] # remove first temp point
    points = np.asarray(points)*scaleGridSizeToMm
    
    trueDensity = len(points)/(domainSizeMm*domainSizeMm)
    
#    plt.scatter(points[:,0],points[:,1],c='k',s=10)
#    plt.axis('square')
    
    r_J, J, SCD, NN = Jfunction(points,domainSizeMm,1)    
    g_average, radii = pairCorrelationFunction_2D_edgeEffectsParallel(points[:,0], points[:,1], domainSizeMm, 1, dr)    
    return lengthScaleMm, trueDensity, rho, np.min(J), r_J[-1], np.max(g_average), it

def GenerateRandomFieldAndReturnOutputStatistics_RhoDens(GRF, rho, targetDensity, domainSizeMm, scaleMmToGridSize,scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr):
    # Uses the "density comparison" method to generate point cloud
    # Under this method, "rho" isn't a probability, it's the ratio of the densities in the tumour and stroma
    import time    
    gridSize = np.shape(GRF)[0]
    
    # How many cells do we have to place?
    targetNum = targetDensity*domainSizeMm*domainSizeMm
    
    # Calculate relative areas
    stromaProp = sum(sum((GRF > 0)))/(gridSize**2)
    tumourProp = sum(sum((GRF <= 0)))/(gridSize**2)
    # Sanity check - everything is tumour or stroma
    assert(np.isclose(stromaProp + tumourProp,1))
    tumourArea = tumourProp*domainSizeMm**2
    stromaArea = stromaProp*domainSizeMm**2
    
    targetTumour = (targetNum*rho*tumourArea/stromaArea)/(1+(rho*tumourArea/stromaArea))
    targetStroma = targetNum - targetTumour
    
    # Check our maths works out
    targetDensityTumour = targetTumour/tumourArea
    targetDensityStroma = targetStroma/stromaArea
    assert(np.isclose(rho,targetDensityTumour/targetDensityStroma)) 
    
    # Great, now lets pick points
    points = [[-1, -1]] # Add temporary point to be removed later
    numPoints = 0
    timeout = time.time() + 10 # Give it up to 10 seconds of being stuck in the loop
    
    numStroma = 0
    numTumour = 0
    while (numStroma < targetStroma or numTumour < targetTumour) and time.time() < timeout:
        x = np.random.rand()*(gridSize-1)
        y = np.random.rand()*(gridSize-1)
        isStroma = GRF[round(x),round(y)] > 0
        if isStroma and numStroma < targetStroma:
            # Point is acceptable - now check that it's not within exclusionLengthscaleMm of another cell
            pointsArr = np.asarray(points)
            diffSquared = (pointsArr[:,0] - y)**2 + (pointsArr[:,1] - x)**2
            if not any(diffSquared < (exclusionLengthscaleMm*scaleMmToGridSize)**2):
                points.append([y,x])
                numStroma = numStroma + 1
                numPoints = numPoints + 1
        if (not isStroma) and numTumour < targetTumour:
            # Point is acceptable - now check that it's not within exclusionLengthscaleMm of another cell
            pointsArr = np.asarray(points)
            diffSquared = (pointsArr[:,0] - y)**2 + (pointsArr[:,1] - x)**2
            if not any(diffSquared < (exclusionLengthscaleMm*scaleMmToGridSize)**2):
                points.append([y,x])
                numTumour = numTumour + 1
                numPoints = numPoints + 1

    points = points[1:] # remove first temp point
    points = np.asarray(points)*scaleGridSizeToMm
    
    trueDensity = len(points)/(domainSizeMm*domainSizeMm)
    
#    plt.scatter(points[:,0],points[:,1],c='k',s=10)
#    plt.axis('square')
    
    r_J, J, SCD, NN = Jfunction(points,domainSizeMm,1)  
    g_average, radii = pairCorrelationFunction_2D_edgeEffectsParallel(points[:,0], points[:,1], domainSizeMm, 1, dr)    
    return lengthScaleMm, trueDensity, rho, np.min(J), r_J[-1], np.max(g_average), it

def AnalyseRandomPointClouds(densityRange,lengthScalerange,gridSize,domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm,it,exclusionLengthscaleMm,dr):
    
    lengthScaleMm = np.random.rand()*(lengthScalerange[1]-lengthScalerange[0]) + lengthScalerange[0]
    density = np.random.randint((densityRange[1]-densityRange[0])) + densityRange[0]
    degreeOfNoise = np.random.rand() # between 0 and 1
    
    GRF = generateGRF(gridSize,domainSizeMm,lengthScaleMm)
    output = GenerateRandomFieldAndReturnOutputStatistics(GRF, degreeOfNoise, density, domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr)
    return output

def AnalyseRandomPointCloudsWithSpecifiedEta(densityRange,lengthScalerange,gridSize,domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm,it,exclusionLengthscaleMm,dr,degreeOfNoise):
    
    lengthScaleMm = np.random.rand()*(lengthScalerange[1]-lengthScalerange[0]) + lengthScalerange[0]
    density = np.random.randint((densityRange[1]-densityRange[0])) + densityRange[0]
#    degreeOfNoise = np.random.rand() # between 0 and 1
    
    GRF = generateGRF(gridSize,domainSizeMm,lengthScaleMm)
    output = GenerateRandomFieldAndReturnOutputStatistics(GRF, degreeOfNoise, density, domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr)
    return output

def AnalyseRandomPointCloudsWithSpecifiedEta_DensityMethod(densityRange,lengthScalerange,gridSize,domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm,it,exclusionLengthscaleMm,dr,degreeOfNoise):
    
    lengthScaleMm = np.random.rand()*(lengthScalerange[1]-lengthScalerange[0]) + lengthScalerange[0]
    density = np.random.randint((densityRange[1]-densityRange[0])) + densityRange[0]
#    degreeOfNoise = np.random.rand() # between 0 and 1
    
    GRF = generateGRF(gridSize,domainSizeMm,lengthScaleMm)
    output = GenerateRandomFieldAndReturnOutputStatistics_RhoDens(GRF, degreeOfNoise, density, domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr)
    return output

def AnalyseRandomPointCloudsWithSpecifiedEta_TwoMethods(densityRange,lengthScalerange,gridSize,domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm,it,exclusionLengthscaleMm,dr,rho):
    
    lengthScaleMm = np.random.rand()*(lengthScalerange[1]-lengthScalerange[0]) + lengthScalerange[0]
    density = np.random.randint((densityRange[1]-densityRange[0])) + densityRange[0]
    
    GRF = generateGRF(gridSize,domainSizeMm,lengthScaleMm)
    output_Rho = GenerateRandomFieldAndReturnOutputStatistics(GRF, rho, density, domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr)
    # Ensure we make an image with the same true density for fair comparison
    density = output_Rho[1]
    output_RhoDens = GenerateRandomFieldAndReturnOutputStatistics_RhoDens(GRF, rho, density, domainSizeMm,scaleMmToGridSize, scaleGridSizeToMm, lengthScaleMm,it,exclusionLengthscaleMm,dr)
    return output_Rho, output_RhoDens

#%%
def norm(x,mu,sigma):
    return ( 1/(sigma * np.sqrt(2 * np.pi)) ) * np.exp( - (x - mu)**2 / (2 * sigma**2))

def normalise(candidatePDF):
    area = np.trapz(candidatePDF)
    return candidatePDF/area

def GenerateLookupTable(eta,means,SDs,ranges):
    #% Make eta lookup table
    
    # Number of features
    if len(np.shape(means)) == 1:
        n = 1
        means = [means]
        SDs = [SDs]
        ranges = [ranges]
#        tableShape = [len(ranges)]
#    else:
        
    n = len(means)
    tableShape = [len(v) for v in ranges]
    
    tableShape.insert(0,len(eta))
    valsByEta = np.zeros(shape=tableShape)
    
    for index in range(len(eta)):
#        eta_value = eta[index]
#        print(eta_value)
        meanOfVbls = [v[index] for v in means]
        SDOfVbls = [v[index] for v in SDs]
        
        dists = [norm(ranges[v],meanOfVbls[v],SDOfVbls[v]) for v in range(len(ranges))]
                
        vals = np.asarray([v for v in dists[0]])
        if n > 0:
            for ind in range(1,n):
                # NB - if n = 1, this has no effect
                vals = np.multiply.outer(vals,dists[ind])

        # normalize
#        print('Linalg: ' + str(np.linalg.norm(vals)))
#        print('Sum: ' + str(np.sum(vals)))
#        vals = vals/np.linalg.norm(vals)
        vals = vals/np.sum(vals)
        valsByEta[index] = vals
    return valsByEta

def GenerateLookupTable_WithCorrelation(eta,observations, ranges):
    #% Make eta lookup table
#    observations = [y3,y2,y1]
#    ranges = [grange,Frange,Jrange]
    n = len(observations)
    # Number of features
#    if n == 1:
#        observations = [observations]
#        ranges = [ranges]

    tableShape = [len(v) for v in ranges]
    
    tableShape.insert(0,len(eta))
    valsByEta = np.zeros(shape=tableShape)
    # Sample covariance matrix
    cov_mat = np.stack(observations,axis=0)
    sigma = np.cov(cov_mat)
    multScalar = ( 1/(np.sqrt(np.linalg.det(sigma))*(2*np.pi)**(n/2)) )
       
    means = [np.mean(v) for v in observations]
    for rhoInd in range(len(eta)):
        print('rhoInd = ' + str(rhoInd))
        for gind in range(len(ranges[0])):
            g = ranges[0][gind]
            for Find in range(len(ranges[1])):
                F = ranges[1][Find]
                for Jind in range(len(ranges[2])):
                    J = ranges[2][Jind]
                    vec = np.asarray([g - means[0],F - means[1],J - means[2]])
#                    print(vec)
                    p = multScalar * np.exp(-0.5*np.matmul(vec,np.matmul(np.linalg.inv(sigma),np.transpose(vec))))
#                    print(p)
                    valsByEta[rhoInd,gind,Find,Jind] = p
            
    return valsByEta
        

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx





#def APR20_GenerateLogLikelihoodTable(trainingData, rangesToGenerateTable):
#    # trainingData: k by N table of observations, where there are k features per point pattern and N training point patterns
#    # rangeToGenerateTable: k arrays (can be different lengths) which determine the discrete values of each variable (gmax, Fmax, Jmin) at which to calculate the likelihood
#    
#    # rangesToGenerateTable = [g_grid, F_grid, J_grid]
#    # trainingData = np.asarray([df['Rho'],df['maxPCF'],df['maxSCDr'],df['minJ']])
#    
#    
#    # Training data has shape k+1 by N, where the first column is the values of rho and the next k columns are observations of the k statistics
#    rhoValues = np.unique(trainingData[0,:])
#    
#    
#    
#    outputShape = [len(v) for v in rangesToGenerateTable]
#    outputShape.insert(0,len(rhoValues))
#    
#    grid = np.meshgrid(*rangesToGenerateTable, indexing='ij')
#    
#    likelihoodTable = np.zeros(shape=(outputShape))
#    for rhoInd, rho in enumerate(rhoValues):
#        mask = trainingData[0,:] == rho
#        data = trainingData[1:,mask]
#        
#        # Estimate mean vector and covariance matrix
#        meanVector = np.mean(data,axis=1)
#        k = len(meanVector)
#        
#        diffGrid = np.copy(grid)
#        for v in range(k):
#            diffGrid[v] = diffGrid[v] - meanVector[v]
#        
#        if k > 1:
#            # Calculate log likelihood for multivariate normal distribution
#            Sigma = np.cov(data)
#            SigmaInverse = np.linalg.inv(Sigma)
#            
#            muTrans_SigmaInv_mu = 0
#            for i in range(k):
#                for j in range(k):
#                    muTrans_SigmaInv_mu = muTrans_SigmaInv_mu + diffGrid[i]*(diffGrid[j]*SigmaInverse[i,j])
#            
#            logL = -0.5*(muTrans_SigmaInv_mu + np.log(np.linalg.det(Sigma)) + k*np.log(2*np.pi))
#        else:
#            # Just use 1D normal distribution log likelihood
#            std = np.std(data)
#            logL = -0.5 * ( np.log(2*np.pi * std**2) + (diffGrid**2)/(std**2) )
#
#            
#
#        likelihoodTable[rhoInd] = logL
#        
#    return rhoValues, likelihoodTable
#    
def func_negExponential(x, a, b,c):
    # Helper function for plotting a negative exponential curve
    return a*np.exp(-b*x) + c   


def GenerateLogLikelihoodTable_FittedMeanAndSD(trainingData, rangesToGenerateTable, meanCoefficients, SDCoefficients):
    # trainingData: k by N table of observations, where there are k features per point pattern and N training point patterns
    # rangeToGenerateTable: k arrays (can be different lengths) which determine the discrete values of each variable (gmax, Fmax, Jmin) at which to calculate the likelihood
    # meanCoefficients: For estimating mean of normal distribution for given rho using func_negExponential
    
    # rangesToGenerateTable = [g_grid, F_grid, J_grid]
    # trainingData = np.asarray([df['Rho'],df['maxPCF'],df['maxSCDr'],df['minJ']])
    
    
    
    # Training data has shape k+1 by N, where the first column is the values of rho and the next k columns are observations of the k statistics
    rhoValues = np.unique(trainingData[0,:])
    
    
    
    outputShape = [len(v) for v in rangesToGenerateTable]
    outputShape.insert(0,len(rhoValues))
    
    grid = np.meshgrid(*rangesToGenerateTable, indexing='ij')
    
    likelihoodTable = np.zeros(shape=(outputShape))
    for rhoInd, rho in enumerate(rhoValues):
        mask = trainingData[0,:] == rho
        data = trainingData[1:,mask]
        
        k = np.shape(trainingData)[0]-1
        
        # Estimate means using func_negExponential
        meanVector = []
        for i in range(k):
            [a,b,c] = meanCoefficients[i]
            m = func_negExponential(rho, a, b,c)
            meanVector.append(m)
        
        diffGrid = np.copy(grid)
        for v in range(k):
            diffGrid[v] = diffGrid[v] - meanVector[v]
        
        if k > 1:
            # Calculate log liklihood for multivariate normal distribution
            # Estimate covariance matrix
            Sigma = np.cov(data)
            SigmaInverse = np.linalg.inv(Sigma)
            
            muTrans_SigmaInv_mu = 0
            for i in range(k):
                for j in range(k):
                    muTrans_SigmaInv_mu = muTrans_SigmaInv_mu + diffGrid[i]*(diffGrid[j]*SigmaInverse[i,j])
            
            logL = -0.5*(muTrans_SigmaInv_mu + np.log(np.linalg.det(Sigma)) + k*np.log(2*np.pi))
        else:
            # Just use 1D normal distribution log likelihood
            #std = np.std(data)
            [a,b,c] = SDCoefficients[0]
            std = func_negExponential(rho, a, b,c)
            logL = -0.5 * ( np.log(2*np.pi * std**2) + (diffGrid**2)/(std**2) )

            

        likelihoodTable[rhoInd] = logL
        
    return rhoValues, likelihoodTable