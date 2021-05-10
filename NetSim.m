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
        
        x1Long %Group for long time arrays
        x2Long
        tLong
        dx1dt_long
        dx2dt_long
        
        jac_funcs %Group for Jacobian functions
        j_existArray %For each element of the Jacobian (same row and column)
            %This array will store a logical true if that x is in the
            %funtion. For example, if the (1, 2) element of the Jacobain
            %(df1(x1, x2) / dx2) = x2^2, the (1, 2, 2) element of this
            %array would be true and the (1, 2, 1) element would be false.
        
        jDerivs %Group for time derivative of jacobian functions
        jderiv_existArray %Same sort of existArray as above, but this time
            %instead of just the first slice being x1 and the second being
            %x2, there will be four slices in the order of :
            %dx1dt, dx2dt, x1, x2. Note this doesn't hold for more than a 2
            %variable system now - doing this in 3 variables will take a
            %lot of rewriting
            
        det_func %anonymous determinant of the jacobian
        tr_func
        det_var  %variables in the determinant function (1x2 logical arry)
        tr_var
        
        mu_dot  %Growth function derivative
        mu_dot_var  %Does the growth function derivative require t variable?
        mu_dotdot  %Growth function second derivative
        mu_dotdot_var  %Does this require t variable input?
        rhok_dot  %Derivative of rhok in time
        rhok_dot_tVar %Do you need to plug in time to rhok_dot (or just k)?
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
            func = @(t,k)(pi()^2 * k.^2)/(obj.grow_func(t))^2;
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
    
    methods   %Methods to set functions (for optimization)
        
        function obj = presetAll(obj)
            %This function should preset all of the equations, derivatives,
            %and variables necessary to completely eliminate any symbolic
            %evaluation during actual simulations
            
            [obj.jac_funcs, obj.j_existArray] = obj.setJacFuncs();
            [obj.jDerivs, obj.jderiv_existArray] = obj.setJacDerivFuncs();
            [obj.det_func, obj.det_var] = obj.makeDetFunc();
            [obj.tr_func, obj.tr_var] = obj.makeTrFunc();
            [obj.mu_dot, obj.mu_dot_var, obj.mu_dotdot, obj.mu_dotdot_var] ...
                = obj.setGrowDerivs();
            [obj.rhok_dot, obj.rhok_dot_tVar] = obj.set_ddt_rhok();
            obj.dx1dt_long = obj.numericalDeriv(obj.x1Long, obj.tLong);
            obj.dx2dt_long = obj.numericalDeriv(obj.x2Long, obj.tLong);
        end
        
        function [x1t, x2t] = valAtTime(obj, time)
            % Gets interpolated x value at whatever time
            [~, idx] = min(abs(obj.tLong - time));
            x1t = obj.x1Long(idx);
            x2t = obj.x2Long(idx);
        end
        
        function [fxnArray, existArray] = setJacFuncs(obj)
            %This function sets up an array of anonymous functions such
            %that we can use the Jacobian elements really quickly, instead
            %of constantly calling simulated stuff, which takes a long
            %time.
            %Also will set logical array to see which variables are used
            jac = obj.rxn_jac;
            [r, c] = size(jac);
            fxnArray = cell(r,c);
            existArray = zeros(r,c,2,'logical');
            syms x1 x2
            
            for i = 1:r
                for j = 1:c
                    inside = symvar(jac(i,j));
                    fxnArray{i,j} = matlabFunction(jac(i,j));
                    if ismember(x1, inside)
                        existArray(i,j,1) = true;
                    else 
                        existArray(i,j,1) = false;
                    end
                    if ismember(x2, inside)
                        existArray(i,j,2) = true;
                    else
                        existArray(i,j,2) = false;
                    end                    
                end
            end
        end
        
        function val = callJac(obj, loc, x1val, x2val)
            %This is a helper function that calls the evaluation of an
            %element in the Jacobian. loc must be the [r, c] vector
            %corresponding to the location in the jacobian and x1val and
            %x2val should either be scalars or vectors of variable values
            %(will hopefully be vectors so I can vectorize code)
            r = loc(1);
            c = loc(2);
            f = obj.jac_funcs{r,c};
            
            if obj.j_existArray(r,c,1) && obj.j_existArray(r,c,2)                
                val = f(x1val, x2val);
            elseif obj.j_existArray(r,c,1) && ~obj.j_existArray(r,c,2)
                val = f(x1val);
            elseif ~obj.j_existArray(r,c,1) && obj.j_existArray(r,c,2)
                val = f(x2val);
            else
                val = f();
            end
        end
        
        function [fxnArray, deriv_existArray] = setJacDerivFuncs(obj)
            %This function sets up an array of anonymous functions so we
            %can use the time derivatives of the Jacobians really easily
            %instead of using symbols each time, which will take a while
            %for sure.
            jac = obj.rxn_jac;
            [r, c] = size(jac);
            fxnArray = cell(r,c);
            deriv_existArray = zeros(r, c, 4, 'logical');
            syms x1 x2 dx1dt dx2dt
            
            for i = 1:r
                for j = 1:c
                    dJx1 = diff(jac(i,j),x1);
                    dJx2 = diff(jac(i,j),x2);
                    dJdt_sym = dJx1 * dx1dt + dJx2 * dx2dt;
                    inside = symvar(dJdt_sym);
                    
                    if ismember(dx1dt, inside)
                        deriv_existArray(i, j, 1) = true;
                    end
                    if ismember(dx2dt, inside)
                        deriv_existArray(i, j, 2) = true;
                    end
                    if ismember(x1, inside)
                        deriv_existArray(i, j, 3) = true;
                    end
                    if ismember(x2, inside)
                        deriv_existArray(i, j, 4) = true;
                    end
                    
                    fxnArray{i,j} = matlabFunction(dJdt_sym);
                end
            end          
        end
        
        function val = callJacDeriv(obj, loc, dx1dt, dx2dt, x1, x2)
            %This is the same as the callJac function above, just for the
            %jacobian derivative. So, there will be a lot more if-end
            %statements, since we're dealing with 2^4 instead of 2^2
            %variables
            r = loc(1);
            c = loc(2);
            f = obj.jDerivs{r,c};
            vect = obj.jderiv_existArray(r, c, :);
            summ = sum(vect);
            
            if summ == 4
                val = f(dx1dt, dx2dt, x1, x2);
            elseif summ == 3
                if ~vect(1)
                    val = f(dx2dt, x1, x2);
                elseif ~vect(2)
                    val = f(dx1dt, x1, x2);
                elseif ~vect(3)
                    val = f(dx1dt, dx2dt, x2);
                else
                    val = f(dx1dt, dx2dt, x1);
                end
            elseif summ == 2
                if vect(1)
                    if vect(2)
                        val = f(dx1dt, dx2dt);
                    elseif vect(3)
                        val = f(dx1dt, x1);
                    elseif vect(4)
                        val = f(dx1dt, x2);
                    end
                elseif vect(2)
                    if vect(3)
                        val = f(dx2dt, x1);
                    elseif vect(4)
                        val = f(dx2dt, x2);
                    end
                else
                    val = f(x1, x2);
                end
            elseif summ == 1
                if vect(1)
                    val = f(dx1dt);
                elseif vect(2)
                    val = f(dx2dt);
                elseif vect(3)
                    val = f(x1);
                else
                    val = f(x2);
                end
            elseif summ == 0
                val = f();
            end
        end
        
        function [func, vars] = makeDetFunc(obj)
            syms x1 x2
            det_v = det(obj.rxn_jac);
            func = matlabFunction(det_v);
            vars = zeros(1,2,'logical');
            
            if ismember(x1, symvar(det_v))
                vars(1,1) = true;
            end
            if ismember(x2, symvar(det_v))
                vars(1,2) = true;
            end
        end
        
        function val = callDet(obj, x1, x2)
            %This calls the Det function
            summ = sum(obj.det_var);
            if summ == 2
                val = obj.det_func(x1, x2);
            elseif summ == 0
                val = obj.det_func();
            else
                if obj.det_var(1)
                    val = obj.det_func(x1);
                else
                    val = obj.det_func(x2);
                end
            end
        end
        
        function [func, vars] = makeTrFunc(obj)
            syms x1 x2
            tr_v = trace(obj.rxn_jac);
            func = matlabFunction(tr_v);
            vars = zeros(1,2,'logical');
            
            if ismember(x1, symvar(tr_v))
                vars(1,1) = true;
            end
            if ismember(x2, symvar(tr_v))
                vars(1,2) = true;
            end
        end
        
        function val = callTrFunc(obj, x1, x2)
            summ = sum(obj.tr_var);
            if summ == 2
                val = obj.tr_func(x1, x2);
            elseif summ == 0
                val = obj.tr_func();
            else
                if obj.tr_var(1)
                    val = obj.tr_func(x1);
                else
                    val = obj.tr_func(x2);
                end
            end
        end
        
        function [mudot, mudotV, mudotdot, mudotdotV] = setGrowDerivs(obj)
            %This function sets all of the growth derivatives and whether
            %they require a time input (true) or are
            %time-independent (false)
            syms t
            gf_sym = obj.grow_func(t);
            
            md_sym = diff(gf_sym, t);
            mudot = matlabFunction(md_sym);
            if ismember(t, symvar(md_sym))
                mudotV = true;
            else
                mudotV = false;
            end
            
            mdd_sym = diff(md_sym, t);
            mudotdot = matlabFunction(mdd_sym);
            if ismember(t, symvar(mdd_sym))
                mudotdotV = true;
            else
                mudotdotV = false;
            end
        end            
        
        function [d_rhok, d_rhok_var] = set_ddt_rhok(obj)
            %This function sets the anonymous function for the time
            %derivative of rhok and a variable to see if you need to plug
            %in the time component into the growth function
            syms t w
            symfunc = obj.rhok(t,w);
            d_rhok_sym = diff(symfunc, t);
            d_rhok = matlabFunction(d_rhok_sym);
            
            if ismember(t, symvar(d_rhok_sym))
                d_rhok_var = true;
            else
                d_rhok_var = false;
            end            
        end
        
        function val = callRhokDot(obj, time, wavenum)
            %This function automatically calls the evaluatable rhok_deriv
            %function with the appropriate inputs
            if obj.rhok_dot_tVar
                val = obj.rhok_dot(time,wavenum);
            else
                val = obj.rhok_dot(wavenum);
            end
        end
        
        function lhsVal = lhsVG3_new(obj, dVal, time, wavenum)
            %Outputs value for the left-hand side of Van Gorder's Theorem 3
            %Note: assuming d1 = 1 and dVal = d2/d1
            %
            %Also Note: This is hopefully going to do the same thing as the
            %previous lhsVG3 function, just in a different, hopefully
            %faster manner. I removed all of the symbols from any repeated
            %functions, so that should speed it up.
            
            %First, pull the actual values of the variables
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            
            %Then, plug stuff in
            lhsVal = obj.callDet(x1val, x2val) - ...
                (dVal * obj.callJac([1 1], x1val, x2val) + ...
                obj.callJac([2 2], x1val, x2val)) * obj.rhok(time, wavenum)...
                + dVal * (obj.rhok(time, wavenum)).^2;
        end
        
        function rhsVal1 = rhs1_new(obj, dVal, time, wavenum)
            %This will return the value of the first part of the right-hand
            %side of Van Gorder's Theorem 3 (without the max term).
            %
            %This new version will just look at the functions, and not
            %require anything symbolic (that was done before this point)
            
            %First, pull the actual values of the variables
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
                        
            %Set term 1
            if obj.mu_dotdot_var %if t is input in mu_dotdot
                term1 = -1*(obj.mu_dotdot(time) / obj.grow_func(time));
            else
                term1 = -1*(obj.mu_dotdot() / obj.grow_func(time));
            end
            
            %Set first part of term two
            if obj.mu_dot_var %if t is variable in mu_dot
                term2_a = -1*(obj.mu_dot(time) / obj.grow_func(time));
            else
                term2_a = -1*(obj.mu_dot() / obj.grow_func(time));
            end
            
            %Set second part of term three
            term2_b = (1 + dVal)*obj.rhok(time, wavenum) - ...
                obj.callTrFunc(x1val, x2val);
            
            %Put together
            rhsVal1 = term1 + term2_a * term2_b;
        end
        
        function poss1Val = possMaxA_new(obj, ~, time, wavenum)
            %The first possible max value for Van Gorder's Theorem 3 rhs
            %term.
            %
            %Note: this new version doesn't do anything with symbolic
            %functions, since they are what's slowing down evaluation I
            %think
            
            %First, figure out time
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            dx1val = obj.dx1dt_long(idx);
            dx2val = obj.dx2dt_long(idx);
            
            %Then, do the first term
            if obj.mu_dot_var
                term1 = (obj.mu_dot(time)/obj.grow_func(time)) * ...
                    (obj.callJacDeriv([1 2], dx1val, dx2val, x1val, x2val) ...
                    / obj.callJac([1 2], x1val, x2val));
            else
                term1 = (obj.mu_dot()/obj.grow_func(time)) * ...
                    (obj.callJacDeriv([1 2], dx1val, dx2val, x1val, x2val) ...
                    / obj.callJac([1 2], x1val, x2val));
            end
            
            %Calculate d/dt part via quotient rule
            ddt = (obj.callJac([1 2], x1val, x2val) * (obj.callRhokDot(time, wavenum)...
                - obj.callJacDeriv([1 1], dx1val, dx2val, x1val, x2val))...
                - (obj.rhok(time, wavenum) - obj.callJac([1 1], x1val, x2val)) ...
                * obj.callJacDeriv([1 2], dx1val, dx2val, x1val, x2val)) ...
                / (obj.callJac([1 2], x1val, x2val))^2;
            %ddt =(j12*(obj.anonDeriv(obj.rhok,t) - obj.jacDeriv(j11, time))...
                %- (obj.rhok(t,k) - j11)*obj.jacDeriv(j12, time)) / j12^2;
                
            %Now, put it all together
            poss1Val = term1 - obj.callJac([1 2],x1val,x2val) * ddt;
        end
        
        function poss2Val = possMaxB_new(obj, dVal, time, wavenum)
            %The second possible max value for Van Gorder's Theorem 3 rhs
            %term.
            %
            %Note: this new version doesn't do anything with symbolic
            %functions, since they are what's slowing down evaluation I
            %think
            
            %First, figure out time
            [~, idx] = min(abs(obj.tLong - time));
            x1val = obj.x1Long(idx);
            x2val = obj.x2Long(idx);
            dx1val = obj.dx1dt_long(idx);
            dx2val = obj.dx2dt_long(idx);
            
            %Then, do the first term
            if obj.mu_dot_var
                term1 = (obj.mu_dot(time)/obj.grow_func(time)) * ...
                    (obj.callJacDeriv([2 1], dx1val, dx2val, x1val, x2val) ...
                    / obj.callJac([2 1], x1val, x2val));
            else
                term1 = (obj.mu_dot()/obj.grow_func(time)) * ...
                    (obj.callJacDeriv([2 1], dx1val, dx2val, x1val, x2val) ...
                    / obj.callJac([2 1], x1val, x2val));
            end
            
            %Next, figure out the ddt stuff
            ddt = (obj.callJac([2 1], x1val, x2val) * ...
                (dVal * obj.callRhokDot(time, wavenum) - ...
                obj.callJacDeriv([2 2],dx1val, dx2val, x1val, x2val)) - ...
                (dVal*obj.rhok(time, wavenum) - obj.callJac([2 2],x1val,x2val))...
                * obj.callJacDeriv([2 1], dx1val, dx2val, x1val, x2val))...
                / (obj.callJac([2 1], x1val, x2val)^2);
            
            %Finally, put it together properly
            poss2Val = term1 - obj.callJac([2 1], x1val, x2val) * ddt;
        end
        
        function rhsVal = rhsVG3_new(obj, dVal, time, wavenum)
            %This function calculates the right-hand side of Van Gorder's
            %Theorem 3 expression
            %
            %New method doesn't use symbolic anything
            pt1 = obj.rhs1_new(dVal, time, wavenum);
            
            aaa = obj.possMaxA_new(dVal, time, wavenum);
            bbb = obj.possMaxB_new(dVal, time, wavenum);
            
            rhsVal = pt1 + max(aaa, bbb);
        end
        
        function logicalImg = findTuringSpace_new(obj, dVal)
            %This function actually does the analysis using Van Gorder's
            %Theorem 3, and outputs the results for one set of conditions
            %as a logical array (can then become an image)
            %
            %The new version pre-defines all functions so we don't have to
            %do anything symbolically during run-time; only at the
            %beginning.
            
            %First, create long time and concentration arrays
            t_vec = obj.homo_state{1,1};
            conc_vec = obj.homo_state{1,2};
            x1_vec = conc_vec(:,1);
            x2_vec = conc_vec(:,2);
            
            [obj.tLong, obj.x1Long, obj.x2Long] = ...
                obj.longTimeSeries(t_vec, x1_vec, t_vec, x2_vec, 500);
            
            %New step: pre-set all of the functions
            obj = obj.presetAll();
                        
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
                lhs = obj.lhsVG3_new(dVal, time, obj.k_vec);
                rhs = obj.rhsVG3_new(dVal, time, obj.k_vec);
                logicalImg(:,i) = (lhs < rhs);
            end
            
            logicalImg = ~flipud(logicalImg); %to align/color it same as VG
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
        
        function derivVal = numericalDeriv(xlong, tlong)
            %This function solves the numerical time derivative of xlong
            %and tlong. Note: xlong and tlong must be vectors of the same
            %length.
            l = length(xlong);
            derivVal = zeros(1,l);
            
            %first and last values
            derivVal(1) = (xlong(2) - xlong(1)) / (tlong(2) - tlong(1));
            derivVal(l) = (xlong(l) - xlong(l-1)) / (tlong(l) - tlong(l-1));
            
            %middle values
            for i = 2:l-1
                left = (xlong(i) - xlong(i-1)) / (tlong(i) - tlong(i-1));
                right = (xlong(i+1) - xlong(i)) / (tlong(i+1) - tlong(i));
                derivVal(i) = (left + right)/2;
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

