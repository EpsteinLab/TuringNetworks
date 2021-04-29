classdef NetSim
    %This is the class of a Turing Network Simulation
    %   Defining a new object for each simulation of one of the networks
    %   from Scholes et al. on a growing domain. The analysis of this
    %   network will follow from the Van Gorder et al. (2021) paper, where
    %   we search for potential transient instabilities (as well as
    %   asymptotic ones)
    
    properties
        k_params % Array of reaction parameters
        rxn_funcs % Reaction functions we'll study
        grow_type % Type of growth
        grow_param % Numerical value of the s parameter
        d_vals = logspace(-3,3,7); % Array of the diffusion characteristics we may look at
        init_size = 10; % Initial size of the growing domain
        final_size = 300; % The final size of the growing domain
        k_vec = 0.1:0.2:80; % Vector of wavenumbers tested for Van Gorder's Turing Condition
        time_k_plot  % Place to store the time versus wavenumber plot
        base_state
        homo_state
    end
    
    properties (Dependent)
        one_func
        grow_ode
        sym_funcs
        rxn_jac
        num_params
        num_var
        t_final
        grow_func
        rhok
    end
    
    properties %(SetAccess = private)
        timestamp
        x1Long
        x2Long
        tLong
    end
    
    methods   %Constructor and Destructor
        function obj = NetSim(k_params, rxn_funcs, grow_type, grow_param, varargin)
            %Construct an instance of the NetSim class
            %Inputs are as follows:
            %   k_params:   A vector of the reaction parameters you need
            %   rxn_funcs:  Array of the number of reaction functions
            %               (usually 2)
            %   grow_type:  Type of growth being used - must a string of
            %               either 'Linear', 'Exponential', or 'Logistic'
            %               (Note: Logistic will be entered later)
            %   grow_param: Value of the growth parameter (scalar)
            %Variable inputs:
            %   d_vals:     Vector containing the diffusion values that we
            %               (may) test
            %   init_size:  initial size of domain (scalar or first part of
            %               an array)
            %   final_size: final size of domain (scalar or second part of
            %               an array)
            
            %If no arguments, default everything to empty
            if nargin == 0
                %Include default values that will be as blank as possible
                k_params = [];
                rxn_funcs = {};
                grow_type = 'None';
                grow_param = 0;
            end
            
            %Set required parameters
            obj.k_params = k_params;
            obj.rxn_funcs = rxn_funcs;
            obj.grow_type = grow_type;
            obj.grow_param = grow_param;
            
            obj.timestamp = datetime('now'); %time stamp of simulation
            
            %Parse optional parameters
            if length(varargin) == 3
                obj.d_vals = varargin{1};
                obj.init_size = varargin{2};
                obj.final_size = varargin{3};
            elseif length(varargin) == 2
                obj.d_vals = varargin{1};
                if ~isscalar(varargin{2})
                    obj.init_size = varargin{2}(1);
                    obj.final_size = varargin{2}(2);
                elseif varargin{2} > 10
                    obj.final_size = varargin{2};
                end
            elseif length(varargin) == 1
                obj.d_vals = varargin{1};
            end
        end
        
        function delete(obj)
            %Destructor
            fclose(obj);
        end
    end
    
    methods   %All of the get methods
        function func = get.one_func(obj)
            %Turn the array of reactions into one vector of functions
            if obj.num_var == 2
                func = @(x,k)[obj.rxn_funcs{1}(x,k); obj.rxn_funcs{2}(x,k)];
            elseif obj.num_var == 3
                func = @(x,k)[obj.rxn_funcs{1}(x,k); obj.rxn_funcs{2}(x,k),...
                    obj.rxn_funcs{3}(x,k)];
            else
                disp('Incorrect number of variables presented')
                func = @(x,k)0;
            end
        end
        
        function func = get.rhok(obj)
            func = @(t,k)(pi()^2 * k^2)/(obj.grow_func(t))^2;
        end
        
        function func = get.grow_ode(obj)
            %Convert the array of reactions and growth into one function
            %vector. Use this for solving homogeneous state
            syms t
            gD = obj.anonDeriv(obj.grow_func, t);
            growDeriv = matlabFunction(gD);
            if obj.num_var == 2
                func = @(t, x, k)[obj.rxn_funcs{1}(x,k) - ...
                    (growDeriv(t) / obj.grow_func(t)) * x(1);...
                    obj.rxn_funcs{2}(x,k) - ...
                    (growDeriv(t) / obj.grow_func(t)) * x(2)];
            elseif obj.num_var == 3
                func = @(t, x, k)[obj.rxn_funcs{1}(x,k) - ...
                    (growDeriv(t) / obj.grow_func(t)) * x(1);...
                    obj.rxn_funcs{2}(x,k) - ...
                    (growDeriv(t) / obj.grow_func(t)) * x(2);
                    obj.rxn_funcs{3}(x,k) - ...
                    (growDeriv(t) / obj.grow_func(t)) * x(3)];
            else
                disp('Incorrect number of variables presented')
                func = @(t,x,k)0;
            end
        end
        
        function jac = get.rxn_jac(obj)
            %Create symbolic Jacobian expression
            X = sym('x',[1 obj.num_var]);
            jac = jacobian(obj.sym_funcs, X);
        end
        
        function gf = get.grow_func(obj)
            %Generates the growth function used for simulation
            if strcmp(obj.grow_type,'Exponential')
                gf = @(t)obj.init_size * exp(obj.grow_param * t);
            elseif strcmp(obj.grow_type,'Linear')
                gf = @(t)((obj.grow_param * t) + 1)*obj.init_size;
            elseif strcmp(obj.grow_type,'Logistic')
                disp('Logistic Growth Not Yet Implemented')
                disp('Default to Zero Growth Function')
                gf = @(t)0;
            else
                disp('No growth function found')
                disp('Default to Zero Growth Function')
                gf = @(t)0;
            end
        end
        
        function val = get.t_final(obj)
            %Calculates the amount of time a simulation needs to run
            syms t
            val = vpasolve(obj.final_size == obj.grow_func(t), t, [0 Inf]);
            val = eval(val);
            if isempty(val) % if no solution is found (logistic growth?)
                val = limit(obj.grow_func(t),t,Inf);
                disp('No exact final time found, using limit as t -> Inf')
            end
        end
        
        function val = get.num_params(obj)
            val = length(obj.k_params);
        end
 
        function val = get.num_var(obj)
            val = length(obj.rxn_funcs);
        end
            
        function fxns = get.sym_funcs(obj)
             %Create symbolic reaction functions (for ease of viewing)
            X = sym('x',[1 obj.num_var]);
            K = sym('k',[1 obj.num_params]);
            fxns = cell(obj.num_var, 1);
            for i = 1:obj.num_var
                fxns{i} = obj.rxn_funcs{i}(X,K);
            end
        end
    end
    
    methods   %Functions to use Van Gorder's Theorem 3
        function lhsVal = lhsVG3(obj, dVal, time, wavenum)
            %Outputs value for left-hand side of Van Gorder's Theorem 3
            %Note: assuming d1 = 1 and dVal = d2/d1
            syms t k x1 x2
            jac = obj.rxn_jac;
            f = det(jac) - (dVal * jac(1,1) + jac(2,2)) * obj.rhok(t,k)...
                + dVal * (obj.rhok(t,k))^2;
            lhsFunc = matlabFunction(f);
            
            %Pull values of variables
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            
            %Now, test to see what we have to insert
            if nargin(lhsFunc) == 4
                lhsVal = lhsFunc(wavenum, time, x1val, x2val);
            elseif nargin(lhsFunc) == 3
                if ismember(x1, symvar(f))
                    lhsVal = lhsFunc(wavenum, time, x1val);
                elseif ismember(x2, symvar(f))
                    lhsVal = lhsFunc(wavenum, time, x2val);
                else
                    disp('Unexpected variable encountered in lhs')
                end
            elseif nargin(lhsFunc) == 2
                if ismember(t, symvar(f)) && ismember(k, symvar(f))
                    lhsVal = lhsFunc(wavenum, time);
                else
                    disp('Unexpected variable in 2-var lhs')
                end
            else
                disp('Wrong number of variables in lhs')
            end
        end
        
        function rhsFunc1 = rhs1(obj, dVal)
            %This returns a symbolic variable that is the right-hand side of Van
            %Gorder's Theorem 3 without the max term
            %Same assumption on d values as above
            syms t k x1 x2
            mu = obj.grow_func;
            muDot = obj.anonDeriv(mu, t);
            muDotDot = obj.anonDeriv(muDot, t);
            f = - (muDotDot/mu(t)) - (muDot/mu(t)) * ...
                ((1 + dVal) * obj.rhok(t, k) - trace(obj.rxn_jac));
            rhsFunc1 = f;
            %rhsFunc1 = matlabFunction(f);
        end
        
        function eval = jacDeriv(obj, symfunc, time)
            %This function uses the chain rule to determine the time
            %derivative of one of the Jacobian elements of the reaction
            %functions. In short, we will use the following relationship:
            %
            %\frac{d J_{i,j}}{d t} = \frac{\partial J_{i,j}}{\partial x1} *
            %\frac{\partial x1}{\partial t} + \frac{\partial J_{i,j}}...
            %{\partial x2} * \frac{\partial x2}{\partial t}
            %
            %Note that since we have the x and time as long vectors instead
            %of symbolic functions, I'll have to numerically approximate
            %the derivative
            syms x1 x2
            dJx1 = diff(symfunc, x1);
            dJx2 = diff(symfunc, x2);
            
            %Determine where in the extended time vectors we're looking
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            maxIdx = length(obj.tLong);
            
            %Calculate dx/dt as average of both sides if possible
            if idx > 1 && idx < maxIdx
                left1 = (obj.x1Long(idx) - obj.x1Long(idx-1)) /...
                    (obj.tLong(idx) - obj.tLong(idx-1));
                right1 = (obj.x1Long(idx+1) - obj.x1Long(idx)) /...
                    (obj.tLong(idx+1) - obj.tLong(idx));
                dx1t = (left1 + right1)/2;
                left2 = (obj.x2Long(idx) - obj.x2Long(idx-1)) /...
                    (obj.tLong(idx) - obj.tLong(idx-1));
                right2 = (obj.x2Long(idx+1) - obj.x2Long(idx)) /...
                    (obj.tLong(idx+1) - obj.tLong(idx));
                dx2t = (left2 + right2)/2;
            elseif idx == 1
                dx1t = (obj.x1Long(idx+1)-obj.x1Long(idx)) /...
                    (obj.tLong(idx+1)-obj.tLong(idx));
                dx2t = (obj.x2Long(idx+1)-obj.x2Long(idx)) /...
                    (obj.tLong(idx+1)-obj.tLong(idx));
            elseif idx == maxIdx
                dx1t = (obj.x1Long(idx)-obj.x1Long(idx-1)) /...
                    (obj.tLong(idx)-obj.tLong(idx-1));
                dx2t = (obj.x2Long(idx)-obj.x2Long(idx-1)) /...
                    (obj.tLong(idx)-obj.tLong(idx-1));
            end
            
            %Now compile into one function and then output value
            dJdt = matlabFunction(dJx1 * dx1t + dJx2 * dx2t);
            if nargin(dJdt) == 2
                eval = dJdt(x1val, x2val);
            elseif nargin(dJdt) == 1
                if isequal(symvar(dJx1),x1) || isequal(symvar(dJx2),x1)
                    eval = dJdt(x1val);
                elseif isequal(symvar(dJx1),x2) || isequal(symvar(dJx2),x2)
                    eval = dJdt(x2val);
                else
                    disp('Error: not reading Jacobian values properly')
                end
            elseif nargin(dJdt) == 0
                eval = dJdt();
            end
        end
            
        function poss1Val = possMaxA(obj, ~, time, wavenum)
            %The first possible max value for Van Gorder's Theorem 3 rhs
            %term
            syms t k x1 x2
            mu = obj.grow_func;
            muDot = obj.anonDeriv(mu, t);
            j11 = obj.rxn_jac(1,1);
            j12 = obj.rxn_jac(1,2);
            j12Dot = obj.jacDeriv(j12, time);
            
            %Determine where in the extended time vectors we're looking
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            
            %Do what's inside the d/dt using quotient rule
            ddt = (j12*(obj.anonDeriv(obj.rhok,t) - obj.jacDeriv(j11, time))...
                - (obj.rhok(t,k) - j11)*obj.jacDeriv(j12, time)) / j12^2;
            inJ11 = symvar(j11);
            inJ12 = symvar(j12);
            
            %Compile the rest of the experession
            expr = ((muDot / mu) * (j12Dot / j12)) - j12 * ddt;
            
            exprF = matlabFunction(expr);
            
            %Determine what variables to plug in, and then do so
            if nargin(exprF) == 4
                poss1Val = exprF(wavenum, time, x1val, x2val);
            elseif nargin(exprF) == 3
                if ismember(x1, inJ11) || ismember(x1, inJ12)
                    poss1Val = exprF(wavenum, time, x1val);
                elseif ismember(x2, inJ11) || ismember(x2, inJ12)
                    poss1Val = exprF(wavenum, time, x2val);
                else
                    disp('Unexpected variable encountered')
                end
            elseif nargin(exprF) == 2
                if isempty(inJ11) && isempty(inJ12)
                    poss1Val = exprF(wavenum, time);
                else
                    disp('Unexpected lack of dependency on t or k')
                end
            else
                disp('Weird lack of dependency of max part of cond')
            end
        end
        
        function poss2Val = possMaxB(obj, dVal, time, wavenum)
            %The second possible max value for Van Gorder's Theorem 3 rhs
            %term
            syms t k x1 x2
            mu = obj.grow_func;
            muDot = obj.anonDeriv(mu, t);
            j22 = obj.rxn_jac(1,1);
            j21 = obj.rxn_jac(1,2);
            j21Dot = obj.jacDeriv(j21, time);
            
            %Determine where in the extended time vectors we're looking
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            
            %Do what's inside the d/dt using quotient rule
            ddt = (j21*(dVal*obj.anonDeriv(obj.rhok,t) - obj.jacDeriv(j22, time))...
                - (dVal*obj.rhok(t,k) - j22)*obj.jacDeriv(j21, time)) / j21^2;
            inJ22 = symvar(j22);
            inJ21 = symvar(j21);
            
            %Compile the rest of the experession
            expr = ((muDot / mu) * (j21Dot / j21)) - j21 * ddt;
            
            exprF = matlabFunction(expr);
            
            %Determine what variables to plug in, and then do so
            if nargin(exprF) == 4
                poss2Val = exprF(wavenum, time, x1val, x2val);
            elseif nargin(exprF) == 3
                if ismember(x1, inJ22) || ismember(x1, inJ21)
                    poss2Val = exprF(wavenum, time, x1val);
                elseif ismember(x2, inJ22) || ismember(x2, inJ21)
                    poss2Val = exprF(wavenum, time, x2val);
                else
                    disp('Unexpected variable encountered')
                end
            elseif nargin(exprF) == 2
                if isempty(inJ22) && isempty(inJ21)
                    poss2Val = exprF(wavenum, time);
                else
                    disp('Unexpected lack of dependency on t or k')
                end
            else
                disp('Weird lack of dependency of max part of cond')
            end
        end
        
        function rhsVal = rhsVG3(obj, dVal, time, wavenum)
            %This function will output the value of the right-hand side of
            %Van Gorder's Theorem 3 expression. It will just be a value,
            %not a function, since I can't get a function out of the
            %derivatives currently
            
            %Determine where in the extended time vectors we're looking
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            
            %First part:
            pt1 = obj.rhs1(dVal);
            syms x1 x2
            pt1Var = symvar(pt1);
            pt1F = matlabFunction(pt1);
            if nargin(pt1F) == 4
                pt1Val = pt1F(wavenum, time, x1val, x2val);
            elseif nargin(pt1F) == 3
                if ismember(x1, pt1Var)
                    pt1Val = pt1F(wavenum, time, x1val);
                elseif ismember(x2, pt1Var)
                    pt1Val = pt1F(wavenum, time, x2val);
                else
                    disp('Unexpected variable encountered in rhs1')
                end
            elseif nargin(pt1F) == 2
                if ismember(t, pt1Var) && ismember(k, pt1Var)
                    pt1Val = pt1F(wavenum, time);
                else
                    disp('Unexpected variable in 2-var rhs1')
                end
            else
                disp('Wrong number of variables in rhs1')
            end            
            
            %Second part
            aaa = obj.possMaxA(dVal, time, wavenum);
            bbb = obj.possMaxB(dVal, time, wavenum);
            pt2Val = max(aaa, bbb);
            
            %Combine
            rhsVal = pt1Val + pt2Val;
        end           
        
        function logicalImg = findTuringSpace(obj, dVal)
            %This function actually does the analysis using Van Gorder's
            %Theorem 3, and outputs the results for one set of conditions
            %as a logical array (can then become an image)
            
            %First, create long time and concentration arrays
            t_vec = obj.homo_state{1,1};
            conc_vec = obj.homo_state{1,2};
            x1_vec = conc_vec(:,1);
            x2_vec = conc_vec(:,2);
            
            [obj.tLong, obj.x1Long, obj.x2Long] = ...
                obj.longTimeSeries(t_vec, x1_vec, t_vec, x2_vec, 500);
            
            %Next, create the wavenumber array I'll be looking through. At
            %least to start with, will need to look at k from 0.1 to 80 in
            %increments of 0.2 as default (though can set ahead of time
            %too)
            if isempty(obj.k_vec) 
                obj.k_vec = 0.1:0.2:80.1;
            end
            
            %Then, create a logical array and propagate with whether
            %Theorem 3 holds.
            logicalImg = zeros(length(obj.k_vec),length(obj.tLong),'logical');
            
            for i = 1:length(obj.tLong)
                time = obj.tLong(i);
                lhs = obj.lhsVG3(dVal, time, obj.k_vec);
                rhs = obj.rhsVG3(dVal, time, obj.k_vec);
                logicalImg(:,i) = (lhs < rhs);
            end
            
            logicalImg = ~flipud(logicalImg); %to align/color it same as VG
        end
            
    end
    
    methods   %Analysis Functions
        function rxnSteadyStates = solveBaseState(obj)
            %This function solves for the base steady state (U*) in the
            %absense of any growth
            X = sym('x',[1 obj.num_var]);
            SS = vpasolve(obj.one_func(X,obj.k_params)==0,X,[0 Inf]);
            [r, ~] = size(SS.x1);
            rxnSteadyStates = zeros(r,obj.num_var);
            if obj.num_var == 2
                rxnSteadyStates(:,1) = eval(SS.x1);
                rxnSteadyStates(:,2) = eval(SS.x2);
            elseif obj.num_var == 3
                rxnSteadyStates(:,1) = eval(SS.x1);
                rxnSteadyStates(:,2) = eval(SS.x2);
                rxnSteadyStates(:,3) = eval(SS.x3);
            else
                disp('Error: only 2 or 3 variables currently coded')
            end
        end
    
        function cellOut = homogeneousStateSim(obj)
            %This function simulates the evolution of the homogeneous base
            %state U(t) from Van Gorder et al.
            baseMat = obj.solveBaseState();
            [num_sol, ~] = size(baseMat);
            simResults = cell(num_sol, 2);
            for i = 1:num_sol %Runs simulations of each base state U*
                baseState = baseMat(i,:);
                try
                    [t, xout] = ode15s(@(t, x)obj.ode(t, x, obj.grow_ode, obj.k_params),...
                        [0 obj.t_final], baseState);
                catch
                    [t, xout] = ode23s(@(t, x)obj.ode(t, x, obj.grow_ode, obj.k_params),...
                        [0 obj.t_final], baseState);
                end
                simResults{i,1} = t;
                simResults{i,2} = xout;
            end
            %Next, tests output to see if the results are similar.
            if num_sol > 1
                num_similar = 0;
                orderedPair = [];
                for i = 1:num_sol-1
                    for j = i+1:num_sol
                        if similarTimeSeries(simResults{i,1},simResults{i,2},...
                                simResults{j,1},simResults{j,2})
                            num_similar = num_similar + 1;
                            orderedPair = [orderedPair; [i j]];
                        end
                    end
                end
                
                cellOut = cell(num_sol - num_similar, 2);
                count = 0;
                for i = 1:num_sol
                    if ~ismember(i, orderedPair(:,1)) %Always skips first value that's similar
                        cellOut{i-count,1} = simResults{i,1};
                        cellOut{i-count,2} = simResults{i,2};
                    else
                        count = count + 1;
                    end
                end
            elseif num_sol == 1
                cellOut = simResults;
            else
                cellOut = {};
            end
        end
        
    end
    
    methods (Static)       
        function deriv = anonDeriv(func, var)
            %Note: this now only works for a 1variable function, will have
            %to add in second variable (if necessary) later
            %Returns the derivative
            if ~isa(var, 'sym')
                var = sym(t);
            end
            deriv = diff(func,var);
            %g = diff(func,var);
            %deriv = matlabFunction(g);
        end
        
        function [tl, x1l, x2l] = longTimeSeries(t1, x1, t2, x2, varargin)
            %This function will interpolate the two timeseries to be much
            %longer (in order to find numerical derivatives and the like). 
            %
            %t1 & t2 are time vectors corresponding to x1 & x2
            %
            %varargin: 
            %If scalar: number of time points sampled. Default is 1000
            %If string: interpolation method. Default is spline
            
            %First, parse varargin
            interpol = 'spline';
            tp = 1000;
            if ~isempty(varargin)
                for i = 1:length(varargin)
                    if isscalar(varargin{i}) && rem(varargin{i},1)==0
                        tp = varargin{i};
                    elseif isstring(varargin{i})
                        interpol = varargin{i};
                    end
                end
            end
            
            %Next, set interpolation bounds
            tmin = max([min(t1), min(t2)]);
            tmax = min([max(t1), max(t2)]);
            tl = linspace(tmin, tmax, tp);
            
            %Interpolate each time series
            try
                x1l = interp1(t1, x1, tl, interpol);
                x2l = interp1(t2, x2, tl, interpol);
            catch
                x1l = interp1(t1, x1, tl, 'linear');
                x2l = interp1(t2, x2, tl, 'linear');
            end
        end   
        
        function TF = similarTimeSeries(t1, x1, t2, x2, varargin)
            %This function determines whether two time series data points
            %are similar (true) or not (false).
            %
            %t1 & t2 are the times corresponding to x1 & x2, respectively.
            %All should be vectors
            %
            %varargin:
            %if it's a scalar, will assume the pValue
            %if it's a string, will assume we're specifying interpolation
            %method
            
            %First, parse varargin
            pVal = 0.01; %default p value
            interpol = 'spline'; %default interpolation method
            if ~isempty(varargin)
                for i = 1:length(varargin)
                    if isscalar(varargin{i}) && varargin{i} < 1 && varargin{i} > 0
                        pVal = varargin{i};
                    elseif isstring(varargin{i})
                        interpol = varargin{i};
                    end
                end
            end
            
            %Next, set time series to appropriate interpolation bounds
            tmin = max([min(t1), min(t2)]);
            tmax = min([max(t1), max(t2)]);
            trange = linspace(tmin, tmax, 1000);
            
            %Interpolate each time series
            try
                int1 = interp1(t1, x1, trange, interpol);
                int2 = interp1(t2, x2, trange, interpol);
            catch
                int1 = interp1(t1, x1, trange, 'linear');
                int2 = interp1(t2, x2, trange, 'linear');
            end
            
            %Check to see if the time series are "the same"
            diff = (int1 - int2) / mean([int1; int2],2);
            if max(diff) > pVal
                TF = false;
            else
                TF = true;
            end
        end
                           
        function dxdt=ode(t, x, f, k)
            %Note: This is copied from STAR method (Scholes et al.)
            %
            %Simple function that provides necessary inputs for running the ode solvers
            %ode15s, ode23 and ode45.
            %The function and parameter values need to be provided by the user (f, k)
            dxdt = f(t, x, k);
        end
        
        function gf = generateGrowthFunction(init_size, grow_type, grow_param)
            %Generates the growth function used for simulation
            if strcmp(grow_type,'Exponential')
                gf = @(t)init_size * exp(grow_param * t);
            elseif strcmp(grow_type,'Linear')
                gf = @(t)((grow_param * t) + 1)*init_size;
            elseif strcmp(grow_type,'Logistic')
                disp('Logistic Growth Not Yet Implemented')
                disp('Default to Zero Growth Function')
                gf = @(t)0;
            else
                disp('No growth function found')
                disp('Default to Zero Growth Function')
                gf = @(t)0;
            end
        end
        
        function symfunc = generateSymbolicFunctions(num_params,num_var,rxn_funcs)
            %Create symbolic reaction functions (for ease of viewing)
            X = sym('x',[1 num_var]);
            K = sym('k',[1 num_params]);
            symfunc = cell(num_var, 1);
            for i = 1:num_var
                symfunc{i} = rxn_funcs{i}(X,K);
            end                
        end
    end
    
end

